# FP32权重无损压缩：ENEC在910B2上的验证

日期：2026-10-24。当前用户要求先压缩FP32模型权重；不启动整模型训练，不保留整模型BF16计算副本。物理2、3卡获准使用，其他卡上的任务不操作。

## 本次完整结果

| 项目 | 实际结果 |
|---|---:|
| 文本参数/张量/分块 | 26,895,998,464 / 851 / 2265 |
| 原始FP32权重 | 100.195402 GiB |
| 归档加映射JSON | 87.106176 GiB |
| 节省/压缩比 | 13.0637% / 1.150267倍 |
| 作者codec位差后原样回退块 | 26 |
| 原样保存的小张量块 | 305 |
| 保存后全帧回读与原FP32 SHA | 2265/2265通过 |

两卡各验证一半文本张量，覆盖完整、没有重算同一张量冒充全模型。09:26:42确认两卡无活动进程。这里的容量包括原字节回退与JSON映射，不含权重解码工作区、梯度、m/v、激活或通信。证据：[全量汇总](../../evidence/2026-10-24/fp32-enec-full-summary.json)。

## 方法选择

选择[ENEC作者实现](https://github.com/hpdps-group/ENEC)，对应[论文v2](https://arxiv.org/html/2604.03298v2)。该工作提供Ascend 910B2的FP32编码器，不依赖CUDA。其他候选如ndzip、MPC、nvCOMP主要提供CPU/CUDA/SYCL实现，不能直接作为本机NPU编解码器。

固定作者提交`3026adde1ed97f8ae528515ae9de9de60e89176a`。先测试`ENEC-FP32-120-6-3-16`，再依据真实权重指数分布测试作者已有的`ENEC-FP32-122-6-3-16`。两组作者kernel均不改动，分别保存构建清单。自写的仅为ACL宿主包装、分块归档、原始块回退和核验代码。

原论文推荐较旧CANN。本次使用服务器已有CANN9.1.0、aarch64/GCC11；只读源码并在私有目录构建，不执行作者安装/清理脚本、不改变共享环境。此前8.3路径已失效，runtime中的cmake启动脚本带旧shebang，故直接指定现有cmake二进制。

## 输入与精度口径

- 读取现有27B checkpoint的全部文本参数，排除视觉模块和MTP。文本参数数量由实际索引/形状重新统计。
- checkpoint中原本BF16的权重精确扩展为FP32，然后压缩；它们初始低16位为零。NPU操作的是FP32字，不是BF16计算副本。
- 另对若干真实权重块执行8次受控FP32 AdamW更新，使低位不再全零，再测编解码。这是固定合成梯度的codec压力测试，不是27B训练或训练质量验证。
- 保存整数位模式SHA256；不是用浮点容差代替无损。测试+0/-0、次正规数、Inf、NaN payload，发现作者codec对所测特殊字不能普遍逐位恢复。
- 任何块若作者解码位差或编码更大，归档保留原始FP32字节并明确标记。不能把此类回退块称为作者codec通过；最终容量包含它们。

## 数据流及内存限制

按实际张量元素数分配给两卡，最大64 MiB FP32块。每块在NPU完成压缩、merge和解压，再核对原始字节，保存独立新归档。每张卡的`weights.enec.parts`与`report.json`共同定义恢复格式：张量名、形状、元素偏移、归档偏移、存储类型、字节数及哈希。不能单独用作者decompress CLI解码整个拼接归档。

`verify_archive.py`重新读取所有保存帧，在不加载原始权重的情况下解压、核对输入SHA。小张量和回退块直接验证原始字节。主转换过程与重新加载验证分别报告。

这一步是离线checkpoint转换。CPU内存用于读取输入、验证和写文件，未把正在训练的状态卸载到CPU。该harness同时保留NPU原始块、压缩临时区、压缩输出和解压块，**不是在线训练的显存节省实现**。显式ACL分配量与驱动HBM剩余分别记录，不冒称为PyTorch峰值。

ENEC保留FP32的符号/尾数字段，主要压缩指数；不能假设权重一定压到一半。受控更新样本的结果也不能代表成熟训练权重。实际完整归档容量及结果见本次journal/evidence。

## 复现入口

在独立目录放置本目录源码，预先配置现有CANN/CPU PyTorch依赖。仅在当轮授权卡上运行；`run.py`和`verify_archive.py`限制物理ID为2或3，ACL结束不调用设备reset。

```bash
python build.py --source "$AUTHOR_SOURCE" --sdk "$CANN_SDK" \
  --cmake "$CMAKE_BINARY" --variant ENEC-FP32-122-6-3-16
python run.py --model "$MODEL_DIR" --device 2 --mode probe --output "$RUN/probe-card2"
python run.py --model "$MODEL_DIR" --device 2 --mode samples --output "$RUN/samples-card2"
python run.py --model "$MODEL_DIR" --device 2 --mode all --output "$RUN/full-card2"
python verify_archive.py --archive "$RUN/full-card2" --device 2
```

3号卡同样运行，二者分工由张量形状确定，不重复计入总模型。所有输出目录必须为新目录，避免覆盖证据。作者编译链的某次增量重链接触发已有预处理对象的`unknown file type`错误；更新宿主包装时使用`build.py --host-only`，复用已校验的作者kernel静态库，不重写kernel。

不上传权重归档、原始状态、大日志或编译产物。不声称压缩归档已集成到前向/反向、梯度即用即弃或有损优化器中。

`run_executed.py`保留本次实际执行的源码；`run.py`在相同逻辑上把读取后的FP32块改为显式复制，以避免未来样本测试对原生FP32映射张量产生原地修改。此次受控更新只涉及原BF16的大矩阵，转换本来就产生新FP32存储；全量转换不进行权重更新。新复制写法未冒称已重跑全量实验。
