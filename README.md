# Qwen27B Ascend 训练日志

记录我们在Ascend 910B2上学习和验证Qwen后训练的过程。这里维护训练环境、数据划分、可复现配置、实验代码和结果；推理部署与性能调优继续由原推理调优仓库维护。

当前已在两张910B2（物理2/3卡）跑通纯FP32文本模型的2次全参数更新：26,895,998,464参数，ENEC权重无损压缩驻留、4-bit近似m/v、梯度反向消费后释放；无整模型BF16副本、无CPU/NVMe运行时状态卸载。B128的m/v约27.414 GiB，观测到的每卡分配峰值约58.5 GiB，独立短验证样本loss 4.1697→2.8569。新checkpoint保存完成；B64两次首步反向均OOM，当前采用B128作为已跑通配置；测试结束2/3卡已释放。仅证明完整短样本训练容量和执行，不证明长期同质量或吞吐可用。[在线原型和结果](experiments/fp32-online-20261024/README.md)。

前置完成全部文本权重ENEC归档：100.195 GiB降至87.106 GiB（含回退块与映射），2265个保存分块全部回读核验通过。[压缩实验](experiments/fp32-enec-20261024/README.md)。

## 2026-10-24 研究结论

已整理并归档 [Adam 状态显存研究 Stage1–13](research/2026-10-24-adam-state-memory/README.md)。严格逐位一致、长期训练、m/v 替代预算 ≤30 GiB 目前未同时达成；允许近似的 4-bit B128 约 27.406 GiB 是该归档当时的条件设计；后续在线短跑另见上方，长期目标质量仍未证明。有限历史因子重放只在固定窗口、相同显式 FP32 算术下精确；真实成熟小 MLP 状态无损帧仍约 83%，不是 15%。

该研究归档提交只包含已完成的 CPU 研究、失败/更正、轻量源码和 [下一次 benchmark 协议](research/2026-10-24-adam-state-memory/NEXT_BENCHMARK.txt)，归档提交当时没有重跑实验或恢复NPU训练；其后用户另行授权的硬件实验按日志单独记录。原始大归档在研究对话中另行交付，Git 中保留文件名和 SHA256。详见 [当天记录](journal/2026-10-24.md)。

2026-10-24归档了后续[混合精度、权重压缩与纠正位研究](research/adam-fp32-lossless/README.md)。仅做CPU位级检查及现成公开11M模型FP32 AdamW状态的离线编码，所有被测恢复方法均逐位还原；所测完整表示按本模型参数量外推仍约163–174 GiB，未达到m/v合计30 GiB目标。这不是Qwen实测，也不是所有无损算法不可能的证明；没有启动训练或NPU任务。

## 历史LoRA基线结果

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

这次验证了短序列LoRA训练可行。验证集只有8条，尚未据此评估通用能力，该LoRA基线当时未测试全参数训练或更长序列。原有GSM8K推理评测题没有进入训练。

## 阅读入口

- [2026-10-24研究归档与验证记录](journal/2026-10-24.md)
- [后续讨论汇总：混合精度、梯度窗口与30 GiB预算](research/adam-fp32-lossless/discussion-summary.md)
- [纠正位容量研究及可复核脚本](research/adam-fp32-lossless/correction-bit-report.md)

- [首个基线的完整配置、结果与复现方法](experiments/qwen27b-swift-sft/README.md)
- [2026-10-23训练日志](journal/2026-10-23.md)
- [FP32 Adam显存讨论与论文线索](research/2026-10-23-fp32-adam-memory.md)
- [过滤后的逐步指标与证据](evidence/2026-10-23/qwen27b-swift-sft.json)
- [训练入口与参数更新审计](experiments/qwen27b-swift-sft/train_baseline.py)
- [独立数据划分](experiments/qwen27b-swift-sft/prepare_data.py)
- [基线启动参数](experiments/qwen27b-swift-sft/run_baseline.sh)

原权重、数据全文、适配器、完整日志和服务器私有路径留在原运行环境，不上传本仓库。仓库中的脚本通过环境变量指定模型和实验目录。
