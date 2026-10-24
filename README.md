# Qwen27B Ascend 训练日志

记录我们在Ascend 910B2上学习和验证Qwen后训练的过程。这里维护训练环境、数据划分、可复现配置、实验代码和结果；推理部署与性能调优继续由原推理调优仓库维护。

当前全参数训练准备已按用户要求暂停：2/3号卡初始化与双卡通信通过，27B尚未加载或训练；18:45:47确认两卡无活动进程。环境保留，不自动恢复。

当前研究转向降低FP32 Adam显存占用：不做CPU卸载、不降低状态精度，讨论梯度窗口、无损压缩及单组状态能否还原m/v。相关容量是静态预算，尚未进行全参Adam实验，详见[讨论记录](research/2026-10-23-fp32-adam-memory.md)。

## 2026-10-24 研究结论

已整理并归档 [Adam 状态显存研究 Stage1–13](research/2026-10-24-adam-state-memory/README.md)。严格逐位一致、长期训练、m/v 替代预算 ≤30 GiB 目前未同时达成；允许近似的 4-bit B128 约 27.406 GiB 仅为条件设计，未证明目标质量/峰值。有限历史因子重放只在固定窗口、相同显式 FP32 算术下精确；真实成熟小 MLP 状态无损帧仍约 83%，不是 15%。

本次只提交已完成的 CPU 研究、失败/更正、轻量源码和 [下一次 benchmark 协议](research/2026-10-24-adam-state-memory/NEXT_BENCHMARK.txt)，没有重跑实验、恢复 NPU 训练或更改精度约束。原始大归档在研究对话中另行交付，Git 中保留文件名和 SHA256。详见 [当天记录](journal/2026-10-24.md)。

## 当前结果

**Qwen3.8-27B已在单张910B2上完成BF16 LoRA训练基线。** 原权重冻结，不量化、不做CPU卸载；这次不是全参数训练。

| 项目 | 结果 |
|---|---|
| 框架 | ms-swift 4.6.0、Transformers 5.14.1、torch-npu 2.10.0.post4 |
| 训练卡 | 单张910B2，物理2号卡 |
| 数据与更新 | 独立32条训练/8条验证；8次更新，batch=1、累积4，最大128 token |
| LoRA | rank=4；q_proj、v_proj、in_proj_qkv；可训练约446万参数 |
| 验证loss | 2.1690 → 1.9409 |
| PyTorch显存峰值 | 分配51.56 GiB，预留51.68 GiB，不含驱动开销 |
| 参数保存 | 160个LoRA张量全部更新，适配器约17.85 MB |
| 结束状态 | 2026-10-23 18:32:12确认自己的训练进程退出，2号卡无活动进程 |

这次验证了短序列LoRA训练可行。验证集只有8条，尚未据此评估通用能力，也未测试全参数训练或更长序列。原有GSM8K推理评测题没有进入训练。

## 阅读入口

- [首个基线的完整配置、结果与复现方法](experiments/qwen27b-swift-sft/README.md)
- [2026-10-23训练日志](journal/2026-10-23.md)
- [FP32 Adam显存讨论与论文线索](research/2026-10-23-fp32-adam-memory.md)
- [过滤后的逐步指标与证据](evidence/2026-10-23/qwen27b-swift-sft.json)
- [训练入口与参数更新审计](experiments/qwen27b-swift-sft/train_baseline.py)
- [独立数据划分](experiments/qwen27b-swift-sft/prepare_data.py)
- [基线启动参数](experiments/qwen27b-swift-sft/run_baseline.sh)

原权重、数据全文、适配器、完整日志和服务器私有路径留在原运行环境，不上传本仓库。仓库中的脚本通过环境变量指定模型和实验目录。
