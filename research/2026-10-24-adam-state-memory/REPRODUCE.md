# 复现与轻量版验证

## 本次提交已做与未做

本次只整理旧记录，并检查 UTF-8、JSON、Python 语法、相对文档链接、内容哈希和原生 C 语法。没有重跑数值实验或启动训练。历史报告里的实测数值来自各阶段原始产物，不是本次发布重新测得。

在本目录执行 `python3 verify_records.py`，只做文件完整性、格式和关键结论一致性检查，不运行研究脚本，不申请网络或设备。

## 环境和限制

原实验环境记录为 Python 3.12.14、NumPy 2.3.5、zlib 1.3.2、Linux x86_64/glibc 2.41；原生 C 使用 GCC 14.2.0。这里只记录版本，不自动安装或升级软件。完整浮点结果取决于平台、归约顺序、舍入、禁用 fast-math/FMA 等条件，不能把其他机器的结果自动称为逐位等价。

原始实验脚本可能覆盖所在目录的 results。要重跑请先复制到独立研究目录；保留本版和原始归档的哈希，不覆盖已提交证据。昂贵脚本不会由 verify_records.py 自动启动。

## 源码与输入

- Stage1–2 的相对布局保留为 adam-state-cpu-20261024/，Stage3–13 保留原兄弟目录，以维持跨阶段导入。
- Stage1–4、Stage6–12 的主要数据通过固定生成器/种子构造。仍应按阶段协议、CLI 和输出目录选择执行，不是所有脚本都可直接无参数运行。
- Stage5/lossless_codecs.py 依赖原始 snapshots；可从相应归档提取，或在单独副本按 collect_snapshots.py 重新生成。Git 只含输入索引，不含原始 .bin/.npy 数据。
- Stage13 编码/独立帧校验依赖归档中的 inputs、results/frames 及部分 provenance 原始状态。Stage13 报告引用的 audit_frame_results.json（约 221 KB）也只在原始归档中，本版保留其他精简审计结果。必须先校验原归档 SHA256，再在单独副本恢复这些文件；不能在缺少它们时把源码语法检查当成 roundtrip 通过。
- Stage11/qwen_audit.py 只依赖已保存的公开 config/官方实现文本计算静态预算；这些源码仅当数据读取，不执行 Transformers。旧仓库完整快照没有复制到本目录，缺少时其校验状态显式为 false。历史 qwen_audit.json 保留原先完成的核对证据。
- 原始 .su 栈表、反汇编、二进制、逐步日志以及被过滤的单配置输出留在归档中。新编译会重新产生 .su，不能复用旧机器的栈开销。

## 原生 C 构建示例（未在本次发布中执行实验）

以下使用原实验编译选项，需已有 GCC 和 NumPy；-mno-red-zone 是 x86 专用。其他架构须另行审核算术/栈计费，不能直接声称复现成功。

Stage9，在 stage9 目录：

```sh
gcc -std=c11 -O2 -fno-fast-math -ffp-contract=off -fno-math-errno -mno-red-zone -fstack-usage -Wall -Wextra frontier.c -lm -o frontier
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_frontier.py
```

Stage10，在 stage10 目录。额外保存 compile_command.txt，避免历史简写命令遗漏这一 run_validation.py 所需文件：

```sh
mkdir -p build/formal
cp factor_replay.c bias_constants.h build/formal/
cd build/formal
printf '%s\n' 'gcc -std=c11 -O2 -fno-fast-math -ffp-contract=off -fno-math-errno -mno-red-zone -fstack-usage -Wall -Wextra -frecord-gcc-switches factor_replay.c -lm -o factor_replay' > compile_command.txt
sh compile_command.txt
cd ../..
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_benchmark.py
```

Stage11 的 run_frontier.py 自行生成各维度构建目录、编译命令与 .su 栈账单；该脚本是完整实验，不是轻量检查。Stage12 的 consolidation.py/controls.py 会重建代数对照与输出。各阶段细节、停止条件和负结果以对应 REPORT.md 为准。

## 证据身份

ARTIFACT_INDEX.json 标识另行交付的原始归档，不提供私有下载 URL。PUBLICATION_PROVENANCE.json 将筛选后的文件对应到原始文件哈希。SHA256SUMS 对应这个 Git 版本，不复用针对完整归档的旧 manifest。
