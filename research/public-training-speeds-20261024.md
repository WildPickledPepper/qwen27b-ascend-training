# Qwen3 系列 27B/32B 公开训练速度资料

整理日期：2026-10-24，Asia/Shanghai。目的：为双卡 Qwen 27B 全参数训练寻找外部速度参照。本文记录公开作者报告及其配置，没有重新运行这些外部实验，也没有启动本地或服务器训练。

## 阅读口径

- 全参数训练、LoRA、推理分别记录；BF16 全参训练不等于纯 FP32 全参训练。
- 卡数使用实际 GPU 数；2 台机器不等于 2 张卡，TP=2 也不代表整个任务只有 2 张卡。
- 更新步可能包含多次梯度累积；总墙钟时间可能包含评估、保存和初始化。不能混用。
- token slots/s 指按批量乘配置序列长度折算的吞吐，不一定是实际非 padding token 吞吐。
- 截至本轮检索，尚未找到可核实的“2 GPU + Qwen3 系列稠密 27B/32B 全参训练 + 数值速度”外部记录。这是检索结果，不表示此类结果不存在。

## 双 GPU：找到的具体速度属于 LoRA

来源：[训练者的 Qwen3.6-27B 实验记录](https://github.com/kreuzhofer/dgx-manager/blob/main/docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md)。

模型 Qwen3.6-27B；BF16 底模；DeepSpeed ZeRO-3 + PEFT；LoRA rank=16，仅约 10,485,760 个可训练参数，占约 0.039%。两台 DGX Spark，每台一颗 GB10，共两个 GPU；这是跨机器双 GPU，不是同机双卡。

| 配置 | 长度、批量 | 作者报告 |
|---|---|---|
| 双 GPU，单路网络 | seq=256，每卡 batch=1，累积=4，全局 batch=8 | 约 25 秒/更新步；500 步总墙钟约 3 小时 55 分 |
| 双 GPU，两路网络 | 同上 | 500 步总墙钟约 2 小时 35 分 |
| 双 GPU，两路网络，5 步小测 | 每卡 batch=1，累积=1，全局 batch=2 | 冷启动第 1 步 7.85 秒，第 5 步 4.26 秒 |
| 双 GPU，单路网络，5 步小测 | 同上一行 | 冷启动第 1 步约 12 秒，第 5 步约 7 秒 |
| 单 GPU 参照 | seq=256，batch=1，累积=4，全局 batch=4 | 约 5.5 秒/更新步；500 步总墙钟约 58 分钟 |

500 步总墙钟包含中途评估等开销；不将 2 小时 35 分除以 500 当成纯训练步耗时。作者还将小测外推为累积 4 次约 15 秒/步，但该数字是估算，不作为独立实测。作者将跨机器变慢主要归因于 ZeRO-3 权重 all-gather；短测启用第二路网络后约 7→4.26 秒。单/双 GPU 的全局批量不同，不作等工作量加速比。

## 全参数：已查到的多卡结果

以下是此前检索及本轮新增资料的汇总。涉及视觉语言模型时，以来源明确的语言模型全参训练范围为准，不将冻结的视觉编码器写成参与训练。

| 模型 / 工作负载 | 实际硬件 | 关键配置 | 公开速度 | 来源 |
|---|---|---|---|---|
| Qwen3.8-27B，MedPix VQA | 8×H100 80GB | NeMo AutoModel，BF16，FSDP2，全局 batch=8，每卡 batch=1，激活检查点，视觉冻结 | 1.732 秒/步；总计约 1865 tokens/s，每卡约 233.07 | [AutoModel 性能汇总](https://docs.nvidia.com/nemo/automodel/v0.7/performance/performance-summary) / [配方](https://docs.nvidia.com/nemo/automodel/v0.7/recipes-e2e-examples/qwen3-8-27b) |
| Qwen3.8-27B，模拟数据训练基准 | 16×H100 | Megatron Bridge，BF16，seq=4096，MBS=2，GBS=32，TP4/PP4/DP1，SP，无激活重算 | 4.591290 秒/步；每卡 1784.248 token slots/s | [Bridge 官方记录](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/qwen/qwen3.8-27b.html) |
| Qwen3.8-27B，优化训练基准 | 16×GB200 | BF16，seq=4096，MBS=2，GBS=32，TP2/PP1/CP1/DP8，无激活重算 | 1.679460 秒/步；每卡 4877.758 token slots/s；峰值 174.120 GiB | 同下方 Bridge 来源 |
| Qwen3.8-27B，另一预训练配置 | 16×GB200 | BF16，seq=4096，MBS=2，GBS=32，TP4/PP4 | 3.551780 秒/步；每卡 2306.449 token slots/s | 同下方 Bridge 来源 |
| Qwen3.8-27B，MedPix 全参 SFT | 16×GB200 | BF16，TP4/PP4，MBS=4，GBS=32，一层 full recompute，确定性模式 | 24.160490 秒/步；每卡 339.066 token slots/s；100 步测试 | 同下方 Bridge 来源 |
| Qwen3.8-27B，8K MedPix 全参 SFT | 16×GB200 | BF16，TP4/PP2/CP2，MBS=4，GBS=32，packing+recompute，非确定性 | 8.164790 秒/步；每卡 2006.665 token slots/s；20 步测试 | 同下方 Bridge 来源 |
| Kev-27B，选择题分类后训练 | 8×H200 | Qwen3.8-27B 基座，FSDP2，BF16 计算、FP32 master/m/v，全局 batch=128 | 1140 步训练约 16.2 小时；折算约 51.16 秒/更新步、2.50 条/秒 | [训练者模型卡](https://huggingface.co/jaredpalmer/kev-27b#training-procedure) |
| Qwen3.5-27B Text2SQL，全参 SFT（本轮新增） | 16×H200，2 台机器 | BF16，FSDP，seq=512，每卡 batch=2，累积=8，GBS=256 | 17.8 万条样本、1 epoch，3 小时 6 分；折算约 15.95 条/秒 | [训练者模型卡](https://huggingface.co/mahernaija/Qwen3.5-27B-Text2SQL#training-details) |

上表 5 条 Bridge 结果来源均为：[NVIDIA Megatron Bridge Qwen3.8-27B 文档](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/qwen/qwen3.8-27b.html)。模拟数据基准取步骤 11–20 均值；真实 SFT 与模拟数据基准分别看待。官方 nightly 页面会更新，应以链接当时的配置与表格核对。

AutoModel 可追溯配置：[固定提交版本 YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/b203f3f9f9f3140c660942a088ff36bbc3a5e856/examples/vlm_finetune/qwen3_8/qwen3_8_27b.yaml)。

折算公式：Kev 平均步耗时 = 16.2×3600÷1140 ≈ 51.16 秒；样本吞吐 = 128÷51.16 ≈ 2.50 条/秒。Text2SQL 样本吞吐 = 178000÷(3×3600+6×60) ≈ 15.95 条/秒。以上来自作者总时长与计数的算术折算，不是独立 profiler 测量。

## 补充线索：有训练配置，未取得可用速度

- [ArcticTraining：单 B300 训练 Qwen3-32B](https://github.com/snowflakedb/ArcticTraining/blob/main/news/qwen3-32b-dgx-station.md)：全参后训练、最长 136K 序列、TiledMLP/Liger、CPU offload，但采用 BF16 optimizer/master；不能当作我们 FP32 Adam 的同精度结果。该文字页未提取到明确步耗时，不补造数字。
- [LlamaFactory issue #10221](https://github.com/hiyouga/LlamaFactory/issues/10221)：作者报告 Qwen3.5-27B 全参训练比 Qwen3-14B/32B 慢至少约 4 倍；没有明确卡数与绝对训练速度，不计作双卡实测。
- [双 RTX PRO 6000 的 Qwen3.5-27B 项目](https://github.com/guangxiangdebizi/LoRA-qwen3.5-27b-v2)：LoRA/QLoRA，未取得明确训练速度，不能当全参性能。
- [verl：Qwen3-32B GSPO](https://verl.readthedocs.io/en/latest/ascend_tutorial/en/model_support/examples/gspo_optimization_practice.html)：配置为 4 台、每台 16 NPU，不是双卡。
- [VIME：Qwen3-32B Ascend GRPO](https://github.com/vllm-project/vime/issues/165)：16 NPU，不是双卡。

## 我们自己的双 910B2 参照

内部已完成的 Qwen27B 纯 FP32 文本全参短跑，使用 CPU 常驻 master/m/v 与 CPUAdam 流水，后 3 步平均 44.51472 秒/步，末步 42.9802 秒；样本仅 28/31/31/32 tokens，batch=1。它不是长序列生产吞吐，不能和上面的 LoRA、BF16 或不同全局批量按秒数直接排名。

本地数值的原始过滤证据位于工作区 `evidence/2026-10-24/fp32-cpu-master-pipeline-results.json`。该实验实现、报告和证据在本次资料归档时仍是独立待提交的工作；本资料提交不包含它们。

## 本轮结论

双 GPU 27B 的公开具体速度目前取得了 LoRA 记录；尚不足以给出双 GPU 全参训练的“正常秒数”。上述多卡全参结果提供量级与框架参照，但不按卡数线性折算成双卡速度。后续找到有原始配置与日志的新结果，再追加本表。
