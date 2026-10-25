# 公开训练配置与我们的批量选择

核对日期：2026-10-25（Asia/Shanghai）。本次只整理资料和已有实验，没有启动训练。

## 配置对照

| 项目 | 我们当前吞吐最佳短跑 | NVIDIA AutoModel | NVIDIA Megatron Bridge |
|---|---|---|---|
| 模型 | Qwen27B文本主干，26,895,998,464个参数更新 | Qwen3.8-27B，冻结视觉塔 | Qwen3.8-27B，语言模型及视觉merger训练、视觉编码器冻结 |
| 硬件 | 2×Ascend 910B2，64 GiB/卡 | 8×H100，80 GB/卡 | 16×GB200 |
| 全局批量GBS | 32 | 8 | 32 |
| 微批量MBS | 每rank 16 | 每rank 1 | 每模型并行组2 |
| 梯度累积 | 1 | 1 | 2 |
| 数据/模型划分 | 双进程ZeRO-3 | FSDP2；TP/PP/CP均1 | TP2、PP1、CP1、DP8 |
| 序列/数据 | 32条短文本，每步2377个有效输入token、3984个填充后token位置 | MedPix图文，变长 | MockVLM，序列4096 |
| 计算精度 | BF16，敏感部分按实现保留高精度 | BF16 | BF16 |
| 优化器与主权重 | CPU FP32 master/m/v，CPUAdam | YAML为AdamW；不能仅由BF16标签断言全部状态dtype | 此配置页未完整列出各状态dtype与驻留位置，不按默认值补写 |
| 激活重算 | 64层中48层 | 开启 | 不开启 |
| 每次更新耗时 | 27.5756秒，3次更新中的后2次均值 | 1.732秒 | 1.679460秒，步骤11–20均值 |
| 实验性质 | 全参短跑，独立8条验证 | 全参SFT，图文任务 | 模拟数据吞吐基准，非收敛证据 |

公开来源：

- [AutoModel性能表，v0.7](https://docs.nvidia.com/nemo/automodel/v0.7/performance/performance-summary)。表中FSDP列为NR；实际FSDP2来自其链接的固定YAML。
- [AutoModel固定版本YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/b203f3f9f9f3140c660942a088ff36bbc3a5e856/examples/vlm_finetune/qwen3_8/qwen3_8_27b.yaml)：global_batch_size=8、local_batch_size=1、FSDP2Manager、activation_checkpointing=true、torch_dtype=bfloat16、AdamW、冻结视觉塔，并配置1层MTP及0.1损失系数。
- [Megatron Bridge验证页](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/qwen/qwen3.8-27b.html)，取“Benchmark · GB200”记录：融合GDN、无激活重算、峰值分配174.120 GiB/GPU，20步短跑，排除首步编译。nightly页面会变化。
- 本地[融合训练报告](../experiments/training-fusion-20261025/README.md)与[过滤证据](../evidence/2026-10-25/training-fusion-results.json)。

硬件、卡数、数据长度、训练模块与计时协议不同，表格用于核对配置，不据此计算同口径速度排名。公开YAML没有显式配置CPU offload，不因此宣称已审计所有运行时状态位置。

## 为什么每卡2个样本仍能全局32个

GB200配置有16张GPU，TP2意味着每2张共同承载一份模型，形成8个数据并行组。每组一次计算2个样本，8组同时计算16个；累积两轮，再做一次参数更新：

`GBS = MBS × 数据并行组数 × 梯度累积次数 = 2 × 8 × 2 = 32`。

TP组内两张卡处理同一批样本，不能再把样本数乘以TP。我们的配置是`16 × 2 × 1 = 32`；GBS2组则是`1 × 2 × 1 = 2`。

## 我们为何选32而不是2

| 已完成融合组 | 全局样本/更新 | 秒/更新 | 样本/秒 |
|---|---:|---:|---:|
| fusion-g2 | 2 | 25.4962 | 0.07844 |
| fusion-g32-selective | 32 | 27.5756 | 1.16045 |

“每次2个样本，一步更快”只描述步耗时；“每次32个，吞吐更高”描述单位时间处理样本数。32样本组只多约2.08秒，却每次处理16倍样本。Profiler显示CPUAdam仍约10.7秒/更新，较大的批量能摊薄这种参数更新成本。

两组样本长度分布、重算设置不同；样本吞吐比约14.8倍是现有两种配方的观察，不是严格控制变量的batch扩展倍率。增大batch也改变更新频率，不能据此保证达到相同质量所需时间同倍改善。后续优化在相同batch、数据和更新数下比较；CPU更新与反向重叠仍是待研究方向。
