> 历史阶段报告。原始二进制、完整逐步数据和旧交付 manifest 不进入 Git；原报告中的解包/复现说明针对原归档。当前轻量版入口见本研究目录的 README.md、REPRODUCE.md 和 ARTIFACT_INDEX.json。后续更正以 EVIDENCE_MAP.txt 和 DECISION_MEMO.txt 为准。

# Adam 状态显存压缩：独立 CPU 数值验证

**最新：第二阶段已完成并按用户要求暂停，准备迁移到云端工作区。** [第二阶段报告](stage2/REPORT.md)包含m/v消融、三个种子的闭环凸优化toy，以及计入旧/新回退缓冲和选择工作区的30GiB预算保护。首阶段及当前阶段都不是27B训练。下面保留首阶段完整记录；后续方向尚未启动。

2026-10-24，Asia/Shanghai。目标仓库 `WildPickledPepper/qwen27b-ascend-training`。本目录是独立研究交付，不修改原训练基线，不提交或推送。

**结果：尚未找到将 FP32 m/v 从 200.39 GiB 无损降至 30 GiB 的方法。** 已实跑 6 类梯度、15 种状态表示、256 步逐步对照；另实跑 SOLO 式 4/2-bit、ZIP-SR、ZE-EDEN，共 3 个舍入种子。可逆编码逐位通过，但压缩率不够；低位状态容量可达目标，递推误差不满足精确等价。本研究没有训练效果接近的证据。

| 路线 | 实测/预算结果 | 判断 |
|---|---|---|
| byte shuffle + zlib；联合 m XOR v；空间 XOR | 8192 元素随机样本整体约 6.75–6.96 bytes/参数；128 窗口加索引约 7.95–8.20 | 无损，但远高于目标 1.19766 bytes/参数 |
| 两个 uint16 位半部表示一个 FP32 | m/v 共 8 bytes/参数 | 可逐位恢复，不省状态容量 |
| BF16 高值 + BF16 数值残差 | m/v 共 8 bytes/参数；10000 个正 FP32 数中仅 328 个重构逐位一致 | 既非普遍无损，也不省容量 |
| BF16 高值 + FP32 数值残差 | m/v 共 12 bytes/参数；本组测试逐位通过 | 容量变大，不能当压缩 |
| m/v 各 BF16 | 共 4 bytes/参数，约100.20 GiB | 近似，仍超目标 |
| 4/4-bit + 两 FP32 scale/128 | 持久态 1.0625 bytes/参数，26.6144 GiB | 有损；朴素 v 零点产生巨大误差 |
| 4-bit m + 4-bit 对数 v（本地试验变体） | 3 个 FP32 元数据/128，27.3972 GiB | 有损；改善部分零点问题，仍有递推偏移 |
| SOLO 式 4/2-bit | codes 0.75 + metadata 0.09375 bytes/参数，21.1350 GiB | 有损；固定 beta 后稀疏尖峰失真突出 |
| ZIP-SR / ZE-EDEN | 26.6144 GiB，加共享码本128 bytes和全局24 bytes | 有损；本次优于朴素 v 量化，但不能称无损 |
| 只保留 m、只保留归一化更新 | 4 bytes/参数，仍约100.20 GiB | 反例否定其普遍等价性 |
| 归一化更新 + log(v)，两 FP32 | 8 bytes/参数 | 不省容量；log/exp 重构改变位值 |

容量使用仓库既有文本参数数目 **26,895,998,464**。30 GiB 要求约 **6.67969 倍压缩**，相当于每参数 **9.58128 bits**，不是每状态各9.58 bits。表中大模型容量是格式预算，**不是 27B 实测显存**。

## 复现

本机 Python 3.11.9；无 NumPy，全部使用标准库，没有安装依赖。进入本目录运行：

```powershell
python experiment.py --n 256 --steps 256 --out results-v2
python followup.py
python candidates.py
python audit.py
```

默认只使用 CPU。主实验约26秒；followup 包含 tracemalloc 的窗口审计约35秒；候选约11秒，具体时间依机器。`results/` 是保留的首轮结果，`results-v2/` 才是当前脚本对应版本，新增 q4_log 并保存真实 step header。旧轮脚本哈希不同属于已记录的版本差异。

- [experiment.py](experiment.py)：FP32 参考、15 格式、合成梯度、反例、历史梯度重放。
- [results-v2/summary.csv](results-v2/summary.csv)：90 行配置汇总。
- [results-v2/per_step.csv](results-v2/per_step.csv)：23040 个向量步骤的 m、v、更新、权重误差。
- [results-v2/run.json](results-v2/run.json)：固定种子、输入哈希、脚本哈希、环境和反例值。
- [results-v2/memory.csv](results-v2/memory.csv)：实际序列化字节和 Python 编解码堆峰值。
- [results-v2/temporal.csv](results-v2/temporal.csv)：时间差分必须连同旧状态锚点计费。
- [followup.py](followup.py)、[followup-results/summary.json](followup-results/summary.json)：位布局边界、Hidden Ratio、固定窗口存储。
- [candidates.py](candidates.py)、[candidate-results/summary.csv](candidate-results/summary.csv)：独立实现的三种近似候选，54 行配置。
- [candidate-results/per_step.csv](candidate-results/per_step.csv)：13824 个候选向量步骤。
- [audit.json](audit.json)、[SHA256SUMS.json](SHA256SUMS.json)：通过的断言、工作区测量及文件哈希。
- [journal/2026-10-24.md](journal/2026-10-24.md)：阶段结果和阻塞。
- [literature_notes.md](literature_notes.md)：主来源、此前文献核查记录和本地验证的边界。

## 固定的参考语义

本地仓库只有已完成的 LoRA 训练入口，没有已运行的全参数 FP32 Adam 实现/状态可作为逐位基线。因此这里**定义独立数值参考**，不冒称与 torch-npu、DeepSpeed CPUAdam 或融合 Adam 逐位一致。

`F(x)` 表示 IEEE binary32 舍入。输入梯度、m、v、权重均为 FP32；每次乘、加、除、平方根节点单独舍入；不使用 FMA。beta1=0.9、beta2=0.999，lr=1e-4，epsilon=1e-8，decoupled weight decay=0.01。beta 和 `(1-beta)` 分别从十进制常量转 FP32；后者不是从已舍入 beta 再作减法。bias correction 在 binary64 计算 `1-beta**t`，再转 FP32。epsilon 位于平方根外。

```text
m = F(F(F(.9)*m) + F(F(.1)*g))
v = F(F(F(.999)*v) + F(F(.001)*F(g*g)))
d = F(F(m/F(1-.9**t)) / F(F(sqrt(F(v/F(1-.999**t)))) + F(1e-8)))
w = F(F(w*F(1-F(F(lr)*F(wd)))) - F(F(lr)*d))
```

先解码旧状态，再更新 m/v，再用当前 FP32 m/v 计算 d 和权重，最后压缩写回；**下步只能读取压缩状态**。日志中的 m/v 是写回再解码值，d 是 lr 乘法之前的归一化更新。FP32 主权重 w 保留，固定梯度不受 w 反馈影响；不进行网络前向，也没有持久 BF16 模型副本。这明确模拟了有 FP32 主权重的优化器侧累加语义。主权重额外4 bytes/参数不在 m/v 压缩容量内；真实 BF16 参数副本另需2 bytes/参数，梯度、激活、通信和分配器另计。

梯度种子20261024：逐元素随机幅度常数、正负交替、1e-5→1e3→1e-3 尺度跳变、[1,-.9+1e-7]近抵消、1e-7底噪加稀疏100倍尖峰、随机符号且跨尺度高斯梯度。`constant` 表示时间常数而非各元素相等。审计另加全元素相等的强可压缩正例。候选量化舍入种子17/29/43，独立64位 PRNG 状态实际写入头部。

误差同时提供逐位差异数、最大绝对误差、每步相对 L2 最大值。近抵消时参考更新接近零，相对误差会非常大，必须结合绝对误差看。所有“相近”只指数值误差，不代表任务准确率。

## 具体失败反例

1. `m` 相同不能推出 `v`：两步梯度 `[1,0]` 和 `[0,FP32(.9)]` 都得到 m=0.08999999612569809，但 v 分别为0.000999000039882958和0.0008099999977275729，m 逐位相同。
2. 当前更新一致不代表可递推：状态 `(m,v)=(1,1)` 与 `(2,4)` 在 t=1 得到同一个 FP32 更新0.3162277638912201。下步同收 g=.5，在 t=2 得到0.2236347794532776与0.2177700698375702。此为状态空间反例，不声称这两个状态都来自相同零初始化的第一步。
3. BF16 数值高低分解反例：FP32 bits `0x6e5c187b` 重构成 `0x6e5c1880`。两个 BF16 尾数不是一个 FP32 的完整有效位集合；按位保存16+16 bits则可恢复但容量不变。
4. 时间 XOR 差分若依赖上一步完整 m/v，旧态锚点仍占8 bytes/参数。只报差分流大小会漏掉真正持久状态。32步、16元素的精确梯度重放占2048 bytes，原 m/v 只占128 bytes；计算量增长不能让存储的历史消失。
5. 朴素4-bit v 可能变为0，而 m 非0；后续步分母受 epsilon 支配。随机序列归一化更新的最大相对 L2 误差约55775.7，权重绝对误差约25.89，已明确失败。把正 v 的零码抬高只解决一类失真，并不恢复原递推。

## 位布局与重参数化边界

MindSpeed 普通参数复用路径的加32768/拆位/减32768在本次24个边界值上全部恢复原 FP32 字，但 BF16 前向值有4个与 RNE 不同。覆盖低半部0x7fff/0x8000/0x8001、高半部奇偶、正负号、正负零。原字可恢复与前向舍入相同是两项独立要求。`v=-0.0` 的 sign bit 存在但比较 `v<0` 返回 false，`abs(v)`会清除此位；因此借用 v 符号位的确定性路径不能直接与 unsigned v 量化组合。此次只验证这些原语和边界，没有复现其完整优化器集成。[官方源码](https://github.com/Ascend/MindSpeed/blob/master/mindspeed/core/memory/reuse_param/reuse_optimizer.py)

Hidden Ratio 仅在 beta1=beta2 时单独对照。1024步随机梯度，未量化 y 在 beta=.5/.9/.999 下，v 位值分别有662/855/961步不同，更新绝对差最大约1.19e-7，说明实数代数等价不等于 FP32 逐位一致。固定4-bit码本误差明显增大。beta=.5、梯度[1,-.5,1]第二步 m=0、v=.1875，y奇点必须回退；本地用原 y 的4-byte槽暂存 v 并另加1-byte标记后，这三步恢复成功。标量格式预算为9 bytes（m+槽+tag），量化正常态为6 bytes（m+码字节+tag）；此项为格式尺寸分析，未实现压缩张量存储，不把半字节码本标签说成已打包。它仍维护两项信息。[论文](https://arxiv.org/html/2609.35392v1)

## 近似候选与工作区

SOLO 式实现用4-bit signed DE m，2-bit unsigned log v，128块、p=.1分位数、FP32 scale/alpha、log域随机舍入。保持 beta=(.9,.999)，没有采用论文建议的 beta1 调低。全零块scale=0；常值块alpha=1；混合零导致分位数为0时使用1e-30下限。这是明确记录的本地边界策略，不是声称与官方源码逐位一致。混合零块因此会把零变成约1e-30。代码实际打包nibble和2-bit码，没有把int8存储误记为2bit。[论文公式](https://arxiv.org/html/2505.00347v2)

ZIP-SR 与 ZE-EDEN 使用 Dyn4 v 码本及同一个 signed DE m 码本。ZIP-SR 的随机概率匹配**当前**预条件器坐标的期望；下步递推和 m/v 相关误差仍可能偏移。ZE-EDEN 非全零块排除零码并作块内尺度修正，会把混合零变成正数。没有执行第三方代码。[算法来源](https://github.com/nubank/adamw4bit/blob/main/src/adamw4bit/quantization.py)

这里的 ZIP-SR/ZE-EDEN 标签指独立复现的 v 舍入规则：论文完整方案的 m 是 NF4，并包含特定层的随机舍入；本地统一 DE m 是受控简化，不能把本地误差当成论文完整方法的反证。

固定窗口验证：8192参数、128参数窗口、16步，就地更新同一个 bytearray，q4/q4_floor持久态实际8720 bytes=8192码+512 scale+16头部；q4_log为8976 bytes。没有索引和残差，因为每块固定步长。窗口 m/v 的数值 payload 实际1024 bytes，旧/新双窗口2048 bytes；Python对象远大于数值payload，记录的 tracemalloc峰值约45KB还包含对照审计分配，不能写成NPU只用2KB。候选单窗口实测编码解码 Python峰值约10–13KB，见audit.json；动态库分配和进程基础内存不在tracemalloc中。**zlib native workspace未测量、NPU分配器峰值未测量**，因此本研究没有完整设备峰值的实测结论。

1%逐元素相对误差阈值的整块 FP32 回退在本组合成数据几乎退成FP32，达到8.078125 bytes/参数。即使忽略索引、标记和窗口，4/4-bit路线也只允许约1.95%参数回退FP32；SOLO式约4.95%。这不是可用回退配额承诺，实际配额更少。

## 下一实验优先级与限制

1. **最优先：signed m 量化误差与 v 量化误差分开消融**，比较 FP32 m+低位v、低位m+FP32 v，只用于定位，不当作达标容量。当前候选在常数、近抵消和尖峰下都偏移，不能仅调整 v 就假定问题解决。随后在仍满足30GiB预算的小比例异常块保护下，重跑同一梯度历史并加入小型CPU闭环优化任务。
2. 在严格无损路线，只有获准且已存在的真实分层 m/v 小样本才能判断压缩熵是否远低于合成数据。当前仓库只含脚本和LoRA过滤证据，没有真实m/v样本；不启动27B来造样本。未知数据不存在被宣称为“无损不可能”的依据，但任意64bit对不可能注入到固定约9.58bit表示。
3. π-Quant 后续可研究100进制紧致打包与128窗口。已读公开实现：两个相邻同类状态元素一个0..99码，m/v分开编码；int8码+每32个元素FP32 scale为1.25 bytes/参数，约31.31GiB，尚未满足目标。稠密无理数轨道是近似存在性，不是有限码无损。未独立执行π-Quant，不将它列入已验证结果。[源码](https://github.com/RUCAIBox/Pi-Quant/blob/main/torch_optim_AdamW_3.32bit.py)

本次没有加载模型、占用NPU、恢复任何暂停实验、访问远程机器、停止服务、删除文件或推送GitHub。CPU合成数据只能排除伪省内存与错误等价论证，不能证明真实27B训练质量、收敛或双卡可行。
