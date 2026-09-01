# kaldi-fbank

**纯 Rust 实现的 Kaldi 兼容在线 FBank 特征提取器** ——
[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) 的 drop-in 替代。

kaldi-fbank 是 [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
（进而也是 [Kaldi](https://github.com/kaldi-asr/kaldi) 特征提取）FBank 流水线的忠实
Rust 移植：相同的配置项、相同的分帧/窗函数/Mel 滤波器组数学、相同的在线流式 API。

- **纯 Rust**：无 C/C++ 依赖、无 `build.rs` 链接外部库，可跨平台编译。
- **不重复造轮子**：实数 FFT 使用主流库
  [`realfft`](https://crates.io/crates/realfft) /
  [`rustfft`](https://crates.io/crates/rustfft)（预计算 plan、SIMD 加速）；CLI 使用
  [`clap`](https://crates.io/crates/clap)。其余为 Kaldi 特征提取算法的忠实移植。
- **输出一致**：在真实语音上，与 kaldi-native-fbank（C++）逐帧输出的最大绝对误差约
  `1e-5 ~ 3e-4`（仅来自 FFT 实现间的浮点舍入差异；`1e-3` 容差内 604/604 用例通过）。
- **性能更快**：相同输入、相同配置下，计算速度约为 C++ 参考实现的 **3~4.5 倍**
  （33 个基准用例中最快 4.45×；长音频与生产配置集中在 3.0×~4.4×，
  详见 [docs/benchmark.md](docs/benchmark.md)）。

## 用法

```rust
use kaldi_fbank::{FbankOptions, OnlineFbank};

let mut opts = FbankOptions::default();
opts.frame_opts.dither = 0.0;      // 确定性输出（默认 0.00003，与 kaldi-native-fbank 一致；Kaldi 本体默认为 1.0）
opts.mel_opts.num_bins = 80;       // sherpa / whisper 常用配置

let mut fbank = OnlineFbank::new(opts);
fbank.accept_waveform(16000.0, &samples_chunk); // 可流式、分块喂入
// ... 更多分块 ...
fbank.input_finished();

let n = fbank.num_frames_ready();
for i in 0..n {
    let frame: &[f32] = fbank.get_frame(i); // dim 维特征
    // ...
}
```

批处理便捷接口：

```rust
let frames: Vec<Vec<f32>> = kaldi_fbank::compute_fbank(&opts, &samples);
```

## 项目结构

```
src/
  options.rs          # FrameExtractionOptions / MelBanksOptions / FbankOptions
  feature_window.rs   # 分帧、去 DC、预加重、窗函数、dither（Kaldi 兼容 RNG）
  mel.rs              # Mel 滤波器组（Kaldi 风格 + librosa 风格，均支持）
  fft.rs              # realfft/rustfft 封装，输出 kiss_fftr 同款紧凑布局
  feature_fbank.rs    # FbankComputer：单帧 FFT → 功率谱 → Mel → log
  online.rs           # OnlineFbank 在线流式提取 + RecyclingVector
examples/fbank-cli.rs # 与 ref/fbank_cli.cc 同参数的命令行工具（基准/对照用）
ref/fbank_cli.cc      # 基于 kaldi-native-fbank 的 C++ 参考 CLI（仅用于对照/基准）
scripts/              # 一致性矩阵、基准对比、golden 生成脚本
tests/golden_test.rs  # golden 文件 + 单元测试（cargo test）
benches/fbank_bench.rs# criterion 基准（cargo bench）
```

## 与 kaldi-native-fbank 的一致性

参考实现构建方式：

```bash
cd /path/to/kaldi-native-fbank   # https://github.com/csukuangfj/kaldi-native-fbank
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DKALDI_NATIVE_FBANK_BUILD_PYTHON=OFF \
      -DKALDI_NATIVE_FBANK_BUILD_TESTS=OFF -DKALDI_NATIVE_FBANK_ENABLE_CHECK=OFF ..
make -j
```

运行一致性矩阵（短/中/长音频 × 31 种配置 × 5 种分块大小，共 600+ 用例）：

```bash
cargo build --release --example fbank-cli   # 构建 CLI（基准工具）
scripts/run_matrix.sh          # 完整矩阵（含 300s 音频）
scripts/run_matrix.sh --quick  # 快速矩阵（不含 300s）
```

覆盖：bin 数（23/40/80）、snip-edges、全部 7 种窗函数、preemph/DC/能量维度/
log/功率谱/非 2 的幂窗长/VTLN 频率边界/librosa 风格（含 slaney 归一化）/
不同帧移帧长/8kHz、44.1kHz 采样率/极端分块（1 样本一帧）。

## 性能基准

```bash
scripts/bench_compare.sh   # 输出 C++ vs Rust 的 RTF 与加速比
cargo bench --bench fbank_bench
```

结果摘要（本机，300s 16kHz 音频，80 bins，10ms 帧移，完整表格见
[docs/benchmark.md](docs/benchmark.md)）：

| 实现 | 计算 300s 音频耗时 | RTF | FPS |
|------|------------------|-----|-----|
| kaldi-native-fbank (C++) | ~0.15 s | ~0.00048 | ~200k |
| **kaldi-fbank (Rust)** | **~0.045 s** | **~0.00014** | **~650k** |

> 说明：kaldi-native-fbank 的 `Rfft` 每一帧都会重新分配 kiss_fft plan，而
> rustfft 只规划一次并复用，这是主要差距来源；Rust 侧同时避免了逐帧堆分配。

## 测试

共 **130 个测试**，`cargo test --release` 全部通过（clippy 0 警告）：

```bash
cargo test --release   # 全部 130 个测试
cargo clippy --all-targets -- -D warnings
```

覆盖矩阵：

| 类别 | 数量 | 覆盖内容 |
|------|------|---------|
| 单元测试（各模块内） | 96 | 窗函数数值、分帧/帧数公式、反射补边、预处理顺序、mel 尺度往返、三角滤波器性质、HTK-mode、librosa 归一化、FFT vs 朴素 DFT、帕塞瓦尔、RecyclingVector、API 契约（采样率不匹配/结束态 panic）等 |
| 边界场景（`tests/edge_cases.rs`） | 20 | 零长/单样本/恰好一帧、全零/恒定/纯音/极端值信号、帧数公式交叉核对、8k/44.1k、非 2 幂窗、snip_edges 反射、能量维度尺寸、确定性 |
| golden 一致性（`tests/golden_test.rs`） | 7 | **33 种配置 × 3 种采样率信号**与 kaldi-native-fbank（C++）逐帧对比（容差 2e-3，含批量与分块流式两条路径） |
| 属性/模糊（`tests/property_test.rs`） | 6 | 300 次随机信号×随机配置：不 panic、输出有限、确定性；80 次随机分块与批量一致；帧数单调性；功率谱非负 |
| 文档示例 | 1 | lib.rs doc test |

另外 `scripts/run_matrix.sh` 提供与 C++ 参考的端到端一致性矩阵（31 配置 × 5 分块 ×
4 时长 + 8k/44.1k，共 604 用例，`PASS=604 FAIL=0`）。

golden 参考文件由 `scripts/gen_golden.sh` 生成（需本机构建的 C++ 参考 CLI）。

注意：`dither != 0` 时 C++ 参考从系统 `rand()` 取种子（不可复现），因此逐比特对齐
仅在 `dither = 0`（生产标准配置）下有意义；kaldi-fbank 提供固定种子以保证可复现。

## 端侧设备（RK3576 / aarch64）交叉编译与基准

已在 **RK3576（rk3576-0203，4×A53 + 4×A72，Debian 12）** 上完成交叉编译部署与基准。

交叉编译（纯 Rust 依赖，用 rust-lld 静态链接，零动态库依赖）：

```bash
rustup target add aarch64-unknown-linux-musl
RUSTFLAGS="-C linker=rust-lld" cargo build --release --target aarch64-unknown-linux-musl \
    --example fbank-cli
# 产出 target/aarch64-unknown-linux-musl/release/examples/fbank-cli（静态链接，1.6MB）
# 设备上按 scripts/bench_arm.sh 约定的名字放置产物：
#   cp examples/fbank-cli /tmp/knf-arm/fbank_cli_rs
```

端侧实测（C++ 参考在设备上用原生 g++ -O3 编译，为 arm 最优基线；33 个基准用例）：

| 指标 | C++ (arm 原生) | **Rust (aarch64 交叉编译)** |
|------|---------------|---------------------------|
| 600s 音频计算耗时 | ~1.65 s | **~0.46 s** |
| RTF | ~0.00275 | **~0.00077** |
| 帧/秒 | ~36,000 | **~130,000** |
| 加速比 | — | **2.4× ~ 4.5×**（长音频 3.2×~4.4×） |

- 一致性矩阵（设备实测）：30 配置 × 3 音频 = **90/90 PASS**，`max_abs_diff ≤ 3.4e-4`。
- 本机 x86-64 同源 33 用例基准：加速比 **1.82× ~ 4.45×**（长音频与生产配置 3.0×~4.4×）。
- 详细报告见 [docs/benchmark.md](docs/benchmark.md) 与
  [docs/benchmark-arm.md](docs/benchmark-arm.md)；端侧复现脚本
  `scripts/bench_arm.sh`、`scripts/matrix_arm.sh`（配合 `scripts/compare_fbank_plain.py`，
  无需 numpy）。

## 许可证

基于 [Apache License 2.0](LICENSE) 授权。

FBank 算法与 API 移植自
[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)（Apache-2.0），
后者又移植自 Kaldi 的[特征提取代码](https://github.com/kaldi-asr/kaldi)（Apache-2.0）。
本项目是对该算法的独立 Rust 实现。
