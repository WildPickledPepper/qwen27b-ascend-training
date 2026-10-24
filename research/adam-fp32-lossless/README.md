# FP32 Adam 无损存储研究

2026-10-24归档，承接[此前讨论](../2026-10-23-fp32-adam-memory.md)。本目录包含静态推导、CPU位级检查和现成公开历史状态的离线编码结果。没有运行训练或NPU任务；没有达到m/v合计30 GiB目标。

另一个研究系列已归档 [Stage1–13](../2026-10-24-adam-state-memory/README.md)。两个系列使用不同小模型、压缩器和编码头，比例不能直接混作同一实验；共同结论是严格无损的长期30 GiB方案尚未成立。

## 阅读顺序

1. [讨论汇总](discussion-summary.md)：u能否递推、BF16/FP32布局、梯度即弃、激活与工作区预算。
2. [研究笔记及文献边界](research-notes.md)：m/v位冗余、BF16权重与可选主权重、Ascend编码相关工作。
3. [纠正位容量报告](correction-bit-report.md)：精确恢复方法、必要侧信息、实际编码量和限制。

## 可复核的检查

| 脚本 | 输出 | 范围 |
| --- | --- | --- |
| [cpu_scalar_bit_checks.py](cpu_scalar_bit_checks.py) | [JSON](cpu_scalar_bit_checks.json) | 固定标量、FP32位字段、受限位打包、静态容量 |
| [state_reconstruction_checks.py](state_reconstruction_checks.py) | [JSON](state_reconstruction_checks.json) | 相同权重/状态比值的反解碰撞 |
| [correction_bit_study.py](correction_bit_study.py) | [JSON](correction_bit_study.json) | m候选区间、序号编码、合成快照和状态族枚举 |
| [public_state_audit.py](public_state_audit.py) | [JSON](public_state_audit.json) | 现成公开FP32 AdamW张量的完整逐位恢复及编码字节数 |

前三个脚本只依赖NumPy，不联网或加载模型。使用独立CPU环境运行：

```powershell
python cpu_scalar_bit_checks.py
python state_reconstruction_checks.py
python correction_bit_study.py
```

公开样本检查使用PyTorch CPU与zstandard。已记录的环境为PyTorch2.6.0+cpu、NumPy1.26.4、zstandard0.25.0，不要求修改或安装训练/NPU环境。

可选的[下载脚本](download_public_state.py)会下载固定版本的公开checkpoint，校验长度与SHA256，随后用weights_only读取CPU张量；不会执行外部仓库代码或运行网络模型。大文件被Git忽略且不随本目录提交：

```powershell
python download_public_state.py
python public_state_audit.py
```

若文件已存在其他位置，可用`public_state_audit.py --checkpoint PATH`指定；仍强制核对固定SHA256。来源见[清单](public-state/source.json)。

## 结果口径

- 30 GiB是两份状态总预算，对应9.58位/参数；不是每份30 GiB。
- 纠正信息必须连同其依赖的u/v、旧权重、索引和元数据一起计数。
- 外部11M模型按单位参数空间外推的GiB，不是目标Qwen实测。
- CPU逐位还原不证明Ascend融合算术逐位等价，也不证明在线峰值显存或性能。
- 目标模型尚无本地可用m/v快照，不通过启动训练补数据。
