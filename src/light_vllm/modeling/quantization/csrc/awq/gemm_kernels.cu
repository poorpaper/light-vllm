/*
SPDX-License-Identifier: Apache-2.0
Adapted from vLLM v0.10.2 and https://github.com/mit-han-lab/llm-awq
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration}, author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang,
Shang and Dang, Xingyu and Han, Song}, journal={arXiv}, year={2023}
}
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "dequantize.cuh"

#include <cuda_fp16.h>

namespace light_vllm {
namespace awq {

/*
直接 AWQ GEMM 的布局与边界：
- Python 只把 1 <= M <= 255 的输入送到这里；更大的 M 先反量化，再交给 cuBLAS。
- 每个 64-thread CTA 计算至多 16 行、N 列，N 只能是 64 或 128；尾行会显式屏蔽。
- K 维每次推进 32，split-K 网格把部分和写入独立输出面，宿主端最后求和。
- B 沿用 AutoAWQ GEMM 的 INT4 pack order；每个 int32 保存 8 个权重。
这些约束是 checkpoint 兼容和索引正确性的基础，修改线程布局时必须同步验证。
*/
template <int N>
__global__ void __launch_bounds__(64)
    gemm_forward_4bit_cuda_m16nXk32(int G, int split_k_iters,
                                    half* __restrict__ A, int* __restrict__ B,
                                    half* __restrict__ scaling_factors,
                                    int* __restrict__ zeros, int M, int IC,
                                    int OC, half* __restrict__ C) {
  // 模板只实例化两种 CTA 输出宽度。
  assert(N == 64 || N == 128);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);
#else
  static constexpr uint32_t ZERO = 0x0;
  float C_warp[32];
  __shared__ half A_shared[16 * (32 + 8)];
  __shared__ half B_shared[32 * (N + 8)];

  int j_factors1 = ((OC + N - 1) / N);
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  half A_shared_warp[8];
  half B_shared_warp[N / 4];
  for (int j_0_4_init = 0; j_0_4_init < N / 32; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / N;
  bool ld_A_flag =
      (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp +
       threadIdx.x * 8 / 32) < M;  // threadIdx.y is warp_id

  half* A_ptr =
      A +
      (((int)blockIdx_y) / j_factors1 * 16 +
       (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) *
          IC +
      (((int)threadIdx.x) % (32 / 8)) * 8;

  int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
               (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
               (((int)blockIdx_y) % j_factors1) * (N / 8) +
               (((int)threadIdx.x) % (N / 8));

  half* A_shared_ptr = A_shared +
                       ((int)threadIdx.y) * row_stride_warp * (32 + 8) +
                       (((int)threadIdx.x) / (32 / 8)) * (32 + 8) +
                       (((int)threadIdx.x) % (32 / 8)) * 8;

  half* B_shared_ptr = B_shared +
                       ((int)threadIdx.y) * (row_stride / 2) * (N + 8) +
                       (((int)threadIdx.x) / (N / 8)) * (N + 8) +
                       (((int)threadIdx.x) % (N / 8)) * 8;

  int* zeros_ptr = zeros + (((int)blockIdx_y) % j_factors1) * (N / 8) +
                   ((int)threadIdx.x) % (N / 8);

  half* scaling_factors_ptr = scaling_factors +
                              (((int)blockIdx_y) % j_factors1) * N +
                              (((int)threadIdx.x) % (N / 8)) * 8;

  half* C_ptr =
      C +
      static_cast<long long>(blockIdx_z) * M * OC  // blockIdz.x -> split_k dim
      + (((int)blockIdx_y) % j_factors1) * N + ((int)threadIdx.y) * (N / 2) +
      (((int)threadIdx.x) % 4) * 2;

  // 每个 split-K 分片只遍历属于自己的 32-wide K tile。
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale =
        *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
      // 每线程读取一个 packed int32，展开 8 个 FP16，再写入带 8 列偏移的 shared tile。
      uint32_t B_loaded =
          *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);

      // 按 (q - zero) * scale 反量化；half2 指令一次处理两个值。
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
      // 写回 shared memory，供后续 ldmatrix/mma 读取。
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (N + 8)) =
          B_loaded_fp16;
    }
    __syncthreads();

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
            "addr; }\n"
            : "=r"(addr)
            : "l"((void*)((&(A_shared[(k_0_1 * 16)])) +
                          (((((int)threadIdx.x) & 15) * 40) +
                           ((((int)threadIdx.x) >> 4) * 8)))));

        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned*)(A_shared_warp + 0))[0]),
              "=r"(((unsigned*)(A_shared_warp + 0))[1]),
              "=r"(((unsigned*)(A_shared_warp + 0))[2]),
              "=r"(((unsigned*)(A_shared_warp + 0))[3])
            : "r"(addr));
      }

      for (int ax1_0 = 0; ax1_0 < N / 32; ++ax1_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
              "addr; }\n"
              : "=r"(addr)
              : "l"((void*)((&(B_shared[(((k_0_1 * (N * 16 + 128)) +
                                          (((int)threadIdx.y) * (N / 2))) +
                                         (ax1_0 * 16))])) +
                            (((((int)threadIdx.x) & 15) * (N + 8)) +
                             ((((int)threadIdx.x) >> 4) * 8)))));
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];\n"
              : "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[0]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[1]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[2]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
        }
      }
      for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
  #else
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
              "%13};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
              "%13};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

  #endif
      }
    }
  }

  // 将 FP32 累加结果写入当前 split-K 输出面；越过 M 的尾行不写。
  for (int ax1_0_1 = 0; ax1_0_1 < (N / 32); ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 +
                       ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
      if (row_offset < M) {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 +
          local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
#endif
}

__global__ void __launch_bounds__(64)
    dequantize_weights(int* __restrict__ B, half* __restrict__ scaling_factors,
                       int* __restrict__ zeros, half* __restrict__ C, int G) {
  static constexpr uint32_t ZERO = 0x0;
  half B_shared[32 * (128 + 8)];

  half* B_shared_ptr2 = B_shared;

  int N = blockDim.x * gridDim.x;  // 2
  int col = (blockIdx.x * blockDim.x + threadIdx.x);
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index1 = 8 * col + 8 * row * N;
  half* C_ptr2 = C + index1;

  int index2 = col + row * N;
  int* B_ptr2 = B + index2;

  int index3 = col + (int)(row / G) * N;
  int* zeros_ptr2 = zeros + index3;
  int index4 = 8 * col + (int)(row / G) * N * 8;
  half* scaling_factors_ptr2 = scaling_factors + index4;

  uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
  uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
  uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr2);

  uint32_t B_loaded = *(uint32_t*)B_ptr2;
  uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.x)
               : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.x)
               : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.y)
               : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.y)
               : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.z)
               : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.z)
               : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.w)
               : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.w)
               : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

  *(uint4*)B_shared_ptr2 = B_loaded_fp16;

  for (int i = 0; i < 8; ++i) {
    *(C_ptr2 + i) = B_shared[i];
  }
}

}  // namespace awq
}  // namespace light_vllm

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy) {
  int in_c = _kernel.size(0);
  int qout_c = _kernel.size(1);
  int out_c = qout_c * 8;
  int G = in_c / _scaling_factors.size(0);

  int x_thread = thx;
  int y_thread = thy;

  int x_blocks = 1;
  int y_blocks = 1;
  if (thx == 0) {
    x_thread = qout_c;
  }
  if (thy == 0) {
    y_thread = in_c;
  }
  if (thx == 0 && thy == 0) {
    x_thread = 8;
    y_thread = 8;
    x_blocks = (int)(qout_c / 8);
    y_blocks = (int)(in_c / 8);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_scaling_factors));

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto de_kernel = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
  auto scaling_factors =
      reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

  dim3 num_blocks(x_blocks, y_blocks);
  dim3 threads_per_block(x_thread, y_thread);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  light_vllm::awq::dequantize_weights<<<num_blocks, threads_per_block, 0, stream>>>(
      kernel, scaling_factors, zeros, de_kernel, G);

  return _de_kernel;
}

// inputs: [M, IC] FP16；qweight/qzeros 每个 int32 打包 8 个 INT4。
// 直接 kernel 明确支持 1 <= M <= 255；更大的 M 由 Python 后端走反量化 + cuBLAS。

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters) {
  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  if (num_in_feats <= 0 || num_in_feats > 255)
    throw std::invalid_argument("AWQ direct GEMM requires M within [1, 255]");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

  auto options = torch::TensorOptions()
                     .dtype(_in_feats.dtype())
                     .device(_in_feats.device());
  at::Tensor _out_feats =
      torch::empty({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
  int num_out_feats = _out_feats.size(-2);
  int num_out_channels = _out_feats.size(-1);

  auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
  auto scaling_factors =
      reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
  int group_size = num_in_channels / _scaling_factors.size(0);

  if (num_out_channels % 64 != 0)
    throw std::invalid_argument("OC is not multiple of cta_N = 64");
  if (num_out_channels % 8 != 0)
    throw std::invalid_argument("OC is not multiple of pack_num = 8");
  if (group_size % 32 != 0)
    throw std::invalid_argument("Group size should be a multiple of 32");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (num_out_channels % 128 == 0) {
    int j_factors1 = num_out_channels / 128 / 1;
    dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);
    // threadIdx.x: 32
    // threadIdx.y: i_factors[2] * j_factors[2]
    dim3 threads_per_block(32, 2);
    light_vllm::awq::gemm_forward_4bit_cuda_m16nXk32<128>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
            num_in_feats, num_in_channels, num_out_channels, out_feats);
  } else if (num_out_channels % 64 == 0) {
    int j_factors1 = num_out_channels / 64 / 1;
    dim3 num_blocks(1 * (num_out_feats + 16 - 1) / 16 * j_factors1 *
                    split_k_iters);

    // threadIdx.x: 32
    // threadIdx.y: i_factors[2] * j_factors[2]
    dim3 threads_per_block(32, 2);
    light_vllm::awq::gemm_forward_4bit_cuda_m16nXk32<64>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
            num_in_feats, num_in_channels, num_out_channels, out_feats);
  }
  return _out_feats.sum(0);
}
