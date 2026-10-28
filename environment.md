# 运行环境与依赖（参考）

本仓库**不含**环境安装，也不含模型权重与数据；脚本通过环境变量定位资源（如 `MODEL_DIR`、`DATA_DIR`、`CPU_ADAM_EXTENSION`），运行前需自行准备。下表只记录实验时实际使用过的版本，便于复现时对照，不构成安装步骤。

## 硬件

| 项目 | 规格 |
|---|---|
| 加速卡 | Ascend 910B2 ×2（按轮授权使用物理 2/3，2026-10-28 起改用 0/3） |
| CPU 分配 | 每 rank 24 个线程，`--numa-spread` 就近绑核（如 NUMA4/CPU96–119、NUMA6/CPU144–167） |
| 主机内存 | 全参训练、CPU 常驻 FP32 master/m/v 时峰值 RSS 约 303 GiB |

## 软件

| 项目 | 版本 | 记录位置 |
|---|---|---|
| CANN | 9.1.0（aarch64 / GCC 11） | [fp32-enec 实验](experiments/fp32-enec-20261024/README.md) |
| PyTorch | 2.10.0（`+cpu` 构建） | [2026-10-27 日志](journal/2026-10-27.md) |
| torch_npu | 2.10.0.post4；训练环境构建 `5dd8ef3f` | [README 历史 LoRA 表](README.md)、[2026-10-27 日志](journal/2026-10-27.md) |
| DeepSpeed | 0.19.8（ZeRO-3 + CPUAdam 扩展自建） | [授权归档](journal/decisions-log.md) |
| ms-swift | 4.6.0（仅 LoRA 基线） | [README 历史 LoRA 表](README.md) |
| Transformers | 5.14.1（仅 LoRA 基线） | 同上 |
| Python | 训练环境未在仓库固定；CPU 研究用到 3.11.9 与 3.12.14 两套记录 | [CPU 研究说明](research/2026-10-24-adam-state-memory/adam-state-cpu-20261024/README.md)、[REPRODUCE](research/2026-10-24-adam-state-memory/REPRODUCE.md) |

## 环境隔离约定

- 训练环境是独立安装，**不修改**共享推理环境、他人任务与原 checkpoint。环境入口脚本属本机私有文件，不入库。
- `CPU_ADAM_EXTENSION` 指向已构建好的 CPUAdam 扩展，避免重复触发 JIT 编译与旧缓存锁；文档里出现过的“用错 env.sh 被版本断言拦下”就是这条约定的由来。
- 依赖版本差异会改变归约顺序与舍入，跨环境比较必须重跑对照，不能直接套用本仓库数字。

## 不需要 GPU 的部分

显存建模、压缩算法研究、证据整理等实验可在纯 CPU 上运行（例如 `research/2026-10-24-adam-state-memory/`、`experiments/sql-memory-liveness-20261027/` 的 meta 路径），这些目录的 README 会注明“未初始化 NPU”。
