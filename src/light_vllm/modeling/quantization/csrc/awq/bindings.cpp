#include <torch/extension.h>

torch::Tensor awq_dequantize(torch::Tensor qweight, torch::Tensor scales,
                             torch::Tensor qzeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);
torch::Tensor awq_gemm(torch::Tensor inputs, torch::Tensor qweight,
                       torch::Tensor scales, torch::Tensor qzeros,
                       int64_t split_k_iters);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("awq_dequantize", &awq_dequantize);
  module.def("awq_gemm", &awq_gemm);
}
