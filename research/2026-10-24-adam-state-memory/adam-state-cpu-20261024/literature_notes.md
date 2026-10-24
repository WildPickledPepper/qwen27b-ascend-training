# 文献线索与本地实验边界

2026-10-24，Asia/Shanghai。此文件区分外部主张、本地验证和待核查项；所有第三方代码只读，没有安装或执行。

| 主来源 | 可借鉴内容 | 不成立的外推 | 本地状态 |
|---|---|---|---|
| [ZIP-SR / ZE-EDEN 论文](https://arxiv.org/abs/2610.12444)，[源码](https://github.com/nubank/adamw4bit) | 预条件器坐标舍入、零点处理与块内尺度校正 | 当前坐标条件无偏不等于未来Adam递推无偏，也非逐位无损 | 已读摘要和量化源码，独立实现v公式；本地m用DE，论文用NF4及局部随机舍入，**不是完整论文配置复现** |
| [SOLO](https://arxiv.org/html/2505.00347v2)，[源码](https://github.com/MTandHJ/SOLO) | 非对称4/2-bit；unsigned EMA自适应对数编码与随机舍入 | 不应把改变beta1后的结果说成原Adam等价 | 已独立实跑固定(.9,.999)版本，定义零/常值边界；21.135GiB是格式预算 |
| [π-Quant 最终NeurIPS论文](https://proceedings.nips.cc/paper_files/paper/2025/file/f943a0abbfe9fb6e46a01411fc7372c9-Paper-Conference.pdf)，[源码](https://github.com/RUCAIBox/Pi-Quant/blob/main/torch_optim_AdamW_3.32bit.py) | 将两个相邻同类状态元素映射为旋转角码；可进一步真正紧致打包 | 最终定理3.1是任意epsilon近似存在性，非固定有限精度精确覆盖；不能据旧预印本批评最终定理 | 已读公开量化代码。int8角码和每32元素scale使m/v总计1.25bytes/参数，即31.31GiB。未执行本地数值复现；最终定理版本更正来自此前研究文献核查 |
| [TurboAdam固定提交](https://github.com/davidkny22/turboadam/tree/c7aeed06efa4a60717d7f6564daf3d1131365941) | UState联合归一化更新与log v，此前研究核查预算约6.5bits/参数含块元数据 | 小模型短程证据不能证明27B；“一个量”仍有有限编码误差 | 已有审查提供的审查结论：CUDA/Triton专用、训练证据有限。本环境浏览该tree受限，未读取固定提交实现、未执行；保留commit不换成浮动main |
| [Hidden Ratio](https://arxiv.org/html/2609.35392v1) | tied-beta下m和y的重参数化、固定4-bit y码本 | 仍保留m；m=0有奇点；实数等价不等于FP32逐位相同 | 独立实跑未量化与4-bit y；奇点回退3步通过；随机长序列存在舍入偏移 |
| [LCT](https://arxiv.org/html/2610.22179v1) | BF16无损、块格式、融合点运算与共享窗口 | 不证明FP32 m/v能6.68倍压缩，不代表已有Ascend实现 | 已浏览论文，作为格式/执行结构参考，未复现 |
| [可逆学习](https://proceedings.mlr.press/v37/maclaurin15.html) | 反向重建SGD+momentum，保留有限精度运算丢失信息 | 不是Adam单状态替代，也不能省略所保存的舍入信息 | 已读摘要，详细舍入信息结论来自此前研究核查。独立实验仅验证保存全历史梯度可精确重放及其容量代价 |
| [MindSpeed参数复用源码](https://github.com/Ascend/MindSpeed/blob/master/mindspeed/core/memory/reuse_param/reuse_optimizer.py) | BF16参数与残余bit共享FP32主权重存储 | 不压缩m/v；原字恢复不保证BF16前向RNE一致 | 已读源码，独立24边界测试检出4个前向差异，并测试v=-0符号边界 |

严格梯度重放必须重建**历史权重下**的梯度，包含当时batch、随机数、混合精度、梯度累积、通信和裁剪语义。拿当前权重重算旧batch不会得到原梯度。重放也有模型、激活、重建权重和优化器窗口峰值；不能只报一个seed的字节数。本地`replay_test`直接存储每步FP32梯度，因此仅验证确定性重放与其显式历史成本，不声称解决真实模型重放。

本地实验已排除的仅是具体格式/递推/测试序列，不是对所有无损算法的否定。真实状态熵、异常分布、层间差异、实际设备工作区及训练效果仍须分别验证。所有外部任务质量数字留在原论文，不计作本地实验结果。
