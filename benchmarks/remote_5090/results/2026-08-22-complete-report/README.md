# 二次整理交付目录

本目录由 `benchmarks/remote_5090/analyze_complete_journey.py` 从已经下载到本地的原始 JSON 与 Prometheus 快照生成，没有重新运行 GPU 实验。

`analysis/` 包含：

- `summary.json`：最终 Light/vLLM 三轮统计与两组 Chain/Trie 统计；
- `final_metrics.csv`：2/8 req/s 的吞吐、TTFT P50/P90/P95、TPOT P50/P90/P95；
- `01_final_throughput.png`：最终吞吐主图；
- `02_final_latency_percentiles.png`：最终 TTFT/TPOT 分位数主图；
- `03_speculation_workload_dependence.png`：Chain/Trie 的 workload 依赖图。

统计口径是先在每轮 64 个请求内计算分位数，再对三轮同一指标取中位数。完整解释、历史路线与原始数据入口见 `../../OPTIMIZATION_JOURNEY.md`。

仓库本地交付包为 `../light-vllm-complete-report-20260822.tar.gz`，包含报告、统计脚本、主图、关键阶段报告和生成这些结论所用的原始数据。SHA-256 记录在同名 `.sha256` 文件中。
