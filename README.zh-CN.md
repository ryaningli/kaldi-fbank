# kaldi-fbank

> 纯 Rust 实现的 Kaldi 兼容在线 FBank 特征提取器。流式优先的 API、忠实复刻的数学、零 C/C++ 依赖。

[![CI](https://github.com/ryaningli/kaldi-fbank/actions/workflows/ci.yml/badge.svg)](https://github.com/ryaningli/kaldi-fbank/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/kaldi-fbank.svg)](https://crates.io/crates/kaldi-fbank)
[![docs.rs](https://img.shields.io/docsrs/kaldi-fbank.svg)](https://docs.rs/kaldi-fbank)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust: 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/tools/install)
[![Pure Rust](https://img.shields.io/badge/pure%20Rust-zero%20C%2FC%2B%2B%20deps-9cf.svg)](#-特性)

[English](README.md) | **简体中文**

## ✨ 特性

- **纯 Rust** —— 运行时依赖只有 [`realfft`](https://crates.io/crates/realfft) / [`rustfft`](https://crates.io/crates/rustfft)，均为纯 Rust。无 C/C++ 代码、无 `build.rs` 链接、无 cmake —— Rust 工具链到哪它就编到哪，一条命令即可交叉编译出全静态 musl 二进制。
- **忠实复刻 Kaldi 数学** —— [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) FBank 流水线（后者又移植自 [Kaldi](https://github.com/kaldi-asr/kaldi) 的特征提取）的从零 Rust 移植：相同的配置项、相同的分帧 / dither / 预加重 / 加窗 / Mel 滤波器组数学、相同的流式 API 形态。真实语音上与 C++ 参考逐帧对比的最大绝对误差为 `1e-5 ~ 3e-4` —— 仅 FFT 实现间的浮点舍入差异。
- **流式优先** —— `accept_waveform` 接受任意大小的分块（最小 1 个样本），帧一经完整即可读取，`get_frame(i)` 返回借用的 `&[f32]`，无逐帧堆分配。需要无限时长流式？`pop(n)` 释放最旧的帧，内存保持平稳。
- **更快** —— 相同输入、相同配置下，速度约为 C++ 参考实现的 **3~4.5 倍**（33 用例基准；长音频与生产配置集中在 3.0×~4.4×）。差距主因：rustfft 一次规划终身复用，而 kaldi-native-fbank 的 `Rfft` 每帧重新分配 kiss_fft plan。
- **配置项全量保留** —— 7 种窗函数（真枚举，带 `FromStr` / `Display`）、snip-edges 与反射补边、能量维度（`use_energy` / `raw_energy` / `htk_compat`）、log 与线性滤波器组、功率谱与幅度谱、Kaldi/HTK 与 librosa 两种 mel 风格（含 slaney 归一化）、VTLN 频率边界、非 2 的幂窗长、固定种子 dither（输出可复现）。
- **居中分帧（0.2.0 新增）** —— `frame_opts.center = true` 遵循 `torch.stft` / torchaudio / NeMo 的 `center=True` 约定：两端零填充、窗中心对齐帧移网格、帧数与 torchaudio 兼容。这是超出 Kaldi C++ 参考的扩展；配合 librosa/slaney mel 风格即可作为 NeMo `FilterbankFeatures` 前端的直接替代。
- **测试扎实** —— **135** 个 `cargo test` 测试（单元 / 边界 / golden / 属性模糊 / 居中分帧兼容性）+ 对本机构建 C++ 参考的 **604 用例**端到端一致性矩阵：`PASS=604 FAIL=0`。

## 📦 安装

```bash
cargo add kaldi-fbank
```

或直接写入 `Cargo.toml`：

```toml
[dependencies]
kaldi-fbank = "0.1"
```

无任何系统前置依赖 —— 不需要 C 编译器、cmake 或内置的 C 源码，`cargo build` 即全部。

## 🚀 快速上手

### 流式（分块喂入）

```rust
use kaldi_fbank::{FbankOptions, OnlineFbank};

let mut opts = FbankOptions::default();
opts.frame_opts.dither = 0.0;    // 确定性输出（默认 0.00003，与 kaldi-native-fbank
                                 // 一致；Kaldi 本体默认 1.0）
opts.mel_opts.num_bins = 80;     // sherpa / whisper 常用配置

let mut fbank = OnlineFbank::new(opts);
fbank.accept_waveform(16000.0, &chunk_1);  // 音频到达即喂入 ——
fbank.accept_waveform(16000.0, &chunk_2);  // 任意大小分块，最小 1 个样本
fbank.input_finished();                    // 冲刷尾部

for i in 0..fbank.num_frames_ready() {
    let frame: &[f32] = fbank.get_frame(i);  // dim 维特征，借用返回
    // ...
}
```

### 批处理（整段信号一次算完）

```rust
let frames: Vec<Vec<f32>> = kaldi_fbank::compute_fbank(&opts, &samples);
```

### 选择窗函数

```rust
use kaldi_fbank::WindowType;

opts.frame_opts.window_type = WindowType::Hann;            // 默认为 Povey
opts.frame_opts.window_type = "hanning".parse().unwrap();  // FromStr，与 Kaldi 同名
```

## 📐 API 一览

| 条目 | 用途 |
| --- | --- |
| `FbankOptions` | 嵌套配置：`frame_opts`（`FrameExtractionOptions`）+ `mel_opts`（`MelBanksOptions`）+ 能量 / log / 功率开关。`Default` 与 kaldi-native-fbank 对齐。 |
| `OnlineFbank` | 流式提取器 —— `accept_waveform` / `input_finished` / `num_frames_ready` / `get_frame` / `pop`。 |
| `compute_fbank` | 批处理助手 —— 整段信号进，`Vec<Vec<f32>>` 出。 |
| `FbankComputer` | 单帧计算内核（FFT → 功率谱 → Mel → log），可脱离流式封装单独使用。 |

## 🔬 一致性与测试

正确性在两个层级上与 C++ 参考对齐。

**仓库内测试** —— 130 个测试，`cargo test` 全绿（clippy 0 警告）：

| 类别 | 数量 | 覆盖内容 |
| --- | --- | --- |
| 单元测试（各模块内） | 96 | 窗函数数值、分帧/帧数公式、反射补边、预处理顺序、mel 尺度往返、三角滤波器性质、HTK 模式、librosa 归一化、FFT vs 朴素 DFT、帕塞瓦尔、RecyclingVector、API 契约（采样率不匹配 / 结束态 panic） |
| 边界场景（`tests/edge_cases.rs`） | 20 | 零长 / 单样本 / 恰好一帧、全零 / 恒定 / 纯音 / 极端值信号、帧数公式交叉核对、8k / 44.1k、非 2 幂窗、snip-edges 反射、能量维度尺寸、确定性 |
| golden 一致性（`tests/golden_test.rs`） | 7 | 33 种配置 × 3 种采样率信号，与 kaldi-native-fbank（C++）逐帧对比（容差 2e-3），批量与分块流式两条路径均覆盖 |
| 属性/模糊（`tests/property_test.rs`） | 6 | 300 次随机信号 × 随机配置：不 panic、输出有限、确定性；80 次随机分块与批量一致；帧数单调性；功率谱非负 |
| 文档示例 | 1 | `lib.rs` doc test |

**端到端一致性矩阵** —— 31 配置 × 5 分块大小 × 4 时长 + 8 kHz/44.1 kHz = **604 用例**，与 C++ 参考逐帧对比（容差 `1e-3`）：`PASS=604 FAIL=0`。

构建参考实现并运行矩阵：

```bash
cd /path/to/kaldi-native-fbank    # https://github.com/csukuangfj/kaldi-native-fbank
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DKALDI_NATIVE_FBANK_BUILD_PYTHON=OFF \
      -DKALDI_NATIVE_FBANK_BUILD_TESTS=OFF -DKALDI_NATIVE_FBANK_ENABLE_CHECK=OFF ..
make -j

cd /path/to/kaldi-fbank
cargo build --release --example fbank-cli    # 对比中的 Rust 一侧
scripts/run_matrix.sh --quick                # 快速矩阵（不含 300s 音频）
scripts/run_matrix.sh                        # 完整矩阵
```

矩阵覆盖：bin 数（23/40/80）、snip-edges、全部 7 种窗函数、preemph / DC / 能量维度、log / 功率谱 / 非 2 的幂窗长、VTLN 频率边界、librosa 风格（含 slaney 归一化）、不同帧移帧长、8 kHz 与 44.1 kHz 采样率、极端分块（每次 1 个样本）。

> `dither != 0` 时 C++ 参考从系统 `rand()` 取种子（不可复现），因此对齐仅在 `dither = 0`（生产标准配置）下有意义。kaldi-fbank 暴露固定种子，dither 开启时输出依然可复现。

## ⚡ 性能

相同输入、相同配置；C++ 侧以 `Release` / `-O3` 构建。完整表格见 [docs/benchmark.md](docs/benchmark.md)。

### x86-64

| 实现 | 300s 音频计算耗时 | RTF | 帧/秒 |
| --- | --- | --- | --- |
| kaldi-native-fbank (C++) | ~0.15 s | ~0.00048 | ~200k |
| **kaldi-fbank (Rust)** | **~0.045 s** | **~0.00014** | **~650k** |

33 用例基准中的加速比：**1.82× ~ 4.45×**（长音频与生产配置 3.0× ~ 4.4×）。

> 差距来源：kaldi-native-fbank 的 `Rfft` **每帧**重新分配 kiss_fft plan，而 rustfft 只规划一次并复用；Rust 侧同时避免了逐帧堆分配。

### 端侧设备 —— RK3576（4×A53 + 4×A72，Debian 12）

C++ 参考在设备上以 `g++ -O3` 原生编译 —— arm 上的最优基线。

| 指标 | C++（arm 原生） | **Rust（aarch64 交叉编译）** |
| --- | --- | --- |
| 600s 音频计算耗时 | ~1.65 s | **~0.46 s** |
| RTF | ~0.00275 | **~0.00077** |
| 帧/秒 | ~36,000 | **~130,000** |
| 加速比 | — | **2.4× ~ 4.5×** |

设备实测一致性：30 配置 × 3 音频 = **90/90 PASS**，`max_abs_diff ≤ 3.4e-4`。完整报告见 [docs/benchmark-arm.md](docs/benchmark-arm.md)；端侧脚本 `scripts/bench_arm.sh` / `scripts/matrix_arm.sh`（配合 `scripts/compare_fbank_plain.py`，无需 numpy）。

交叉编译全静态二进制（纯 Rust 依赖 + rust-lld，零动态库）：

```bash
rustup target add aarch64-unknown-linux-musl
RUSTFLAGS="-C linker=rust-lld" \
  cargo build --release --target aarch64-unknown-linux-musl --example fbank-cli
# → target/aarch64-unknown-linux-musl/release/examples/fbank-cli（静态，1.6 MB）
```

## 🛠️ 开发

```bash
cargo fmt --all                             # 格式化
cargo clippy --all-targets -- -D warnings   # lint（警告视为错误）
cargo test                                  # 全部 130 个测试
cargo bench --bench fbank_bench             # criterion 基准
```

每次提交前的质量门：fmt 绿、clippy 绿、测试绿。CI（Linux）跑同样三项；推送 `v*` tag 时先校验其与 `Cargo.toml` 版本一致，再发布到 crates.io。

一致性矩阵额外需要本机构建的 C++ 参考 —— 见[🔬 一致性与测试](#-一致性与测试)。

## 🧱 架构

流水线就是 Kaldi 的，逐模块对应：

```
PCM 样本
   │  accept_waveform() —— 分块、流式
   ▼
feature_window.rs     分帧 · dither · 去 DC · 预加重 · 加窗
   ▼
fft.rs                实数 FFT（realfft/rustfft），kiss_fftr 同款紧凑布局
   ▼
feature_fbank.rs      功率谱 → Mel 滤波器组 → log    （FbankComputer）
   ▼
online.rs             OnlineFbank：帧缓存（RecyclingVector），get_frame(i) → &[f32]
```

```
src/
  options.rs           # FrameExtractionOptions / MelBanksOptions / FbankOptions / WindowType
  feature_window.rs    # 分帧、dither（Kaldi 兼容 RNG）、去 DC、预加重、窗函数
  mel.rs               # Mel 滤波器组（Kaldi/HTK 风格 + librosa 风格）
  fft.rs               # realfft/rustfft 封装
  feature_fbank.rs     # FbankComputer：单帧 FFT → 功率谱 → Mel → log
  online.rs            # OnlineFbank 流式 + RecyclingVector
examples/fbank-cli.rs  # 与 ref/fbank_cli.cc 同参数的 CLI（对照/基准工具）
ref/fbank_cli.cc       # 基于 kaldi-native-fbank 的 C++ 参考 CLI（仅用于对照）
scripts/               # 一致性矩阵、基准对比、golden 生成
tests/                 # golden / 边界 / 属性测试
benches/fbank_bench.rs # criterion 基准
```

## 📚 文档

- [docs.rs/kaldi-fbank](https://docs.rs/kaldi-fbank) —— API 参考
- [docs/benchmark.md](docs/benchmark.md) —— x86-64 基准表格
- [docs/benchmark-arm.md](docs/benchmark-arm.md) —— RK3576 端侧基准

## 🤝 参与贡献

欢迎贡献。凡触及数值路径的改动，请保持 `cargo test` 全绿，并尽可能扩展一致性矩阵 —— 与 C++ 参考的输出对齐是本项目的核心契约。

## 📄 许可证

基于 [Apache License 2.0](LICENSE) 授权。

FBank 算法与 API 移植自 [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)（Apache-2.0），后者又移植自 Kaldi 的[特征提取代码](https://github.com/kaldi-asr/kaldi)（Apache-2.0）。本项目是对该算法的独立 Rust 实现。
