# Speculative Decoding in Large Language Models: Adaptive Draft Control and Component-Aware Self-Speculation

## 1. Executive Summary

The two papers address complementary aspects of speculative decoding:

1. **AdaEDL** improves draft-length control in conventional two-model speculative decoding. It is a training-free, parameter-free controller that uses the entropy of currently observed draft-model logits to approximate a lower bound on the expected acceptance probability of the next drafted token. Drafting stops early when the criterion indicates that further drafting is unlikely to be beneficial. Its threshold is dynamically adapted from observed acceptance-rate statistics rather than fixed globally.
2. **Component-Aware Self-Speculative Decoding** uses an internal SSM or linear-attention subgraph of a hybrid language model as a zero-additional-parameter drafter. The full hybrid model verifies the resulting tokens. The reported results show strong architecture dependence: parallel hybrids such as Falcon-H1 have substantially higher acceptance than sequential hybrids such as Qwen3.5.

AdaEDL reports measured throughput improvements of up to **2.21× over autoregressive decoding** on the supplied benchmarks. The component-aware paper reports acceptance, agreement, distributional-distance, and perplexity metrics, but **does not report end-to-end throughput, latency, VRAM usage, or energy measurements** in the supplied facts. Its acceptance results therefore must not be presented as measured speedups.

## 2. Comparative Summary

| Dimension | AdaEDL | Component-Aware Self-Speculative Decoding |
|---|---|---|
| Primary objective | Adaptively stop external drafting | Use an internal hybrid-model component as the drafter |
| Draft source | Separate Llama2-Drafter-115M model | Internal SSM or linear-attention subgraph |
| Additional controller/model parameters | No additional AdaEDL parameters; a separate drafter is still required | No additional drafter parameters |
| Training requirement | Training-free | No additional training is reported in the supplied facts |
| Core signal | Draft-logit entropy and observed acceptance statistics | Agreement between an isolated component and the full hybrid model |
| End-to-end throughput reported | Yes | No |
| Hardware reported | One NVIDIA A100 80 GB, FP32 | Not reported |
| Main limitation | Retains separate-drafter memory and execution costs | Viability is architecture-dependent and systems-level speedup is unverified |

## 3. Verified AdaEDL Throughput Results

All supplied AdaEDL measurements use a **Llama2-7B target**, **Llama2-Drafter-115M draft model**, one **NVIDIA A100 80 GB**, and **FP32**. Speedups are calculated relative to the autoregressive baseline for the same dataset.

| Dataset | Maximum draft length | Autoregressive | Base-SPD | Max-Confidence-SPD | AdaEDL |
|---|---:|---:|---:|---:|---:|
| CNN-DM | 16 | 25.74 TPS | 36.30 TPS (1.41×) | 49.50 TPS (1.92×) | **54.10 TPS (2.10×)** |
| Dolly-15k | 16 | 29.02 TPS | 32.10 TPS (1.11×) | 55.80 TPS (1.92×) | **56.10 TPS (1.93×)** |
| WMT-19 German–English translation | 16 | 29.80 TPS | 22.30 TPS (0.75×) | 43.70 TPS (1.47×) | **43.90 TPS (1.47×)** |
| CNN-DM | 7 | 25.74 TPS | 51.50 TPS (2.00×) | 53.50 TPS (2.08×) | **56.90 TPS (2.21×)** |
| Dolly-15k | 7 | 29.02 TPS | 47.60 TPS (1.64×) | 56.60 TPS (1.95×) | **57.10 TPS (1.97×)** |
| WMT-19 German–English translation | 7 | 29.80 TPS | 32.70 TPS (1.10×) | 45.10 TPS (1.51×) | **45.20 TPS (1.52×)** |
| CNN-DM | 3 | 25.74 TPS | 54.10 TPS (2.10×) | 53.10 TPS (2.06×) | **55.70 TPS (2.16×)** |
| Dolly-15k | 3 | 29.02 TPS | 54.70 TPS (1.88×) | 55.70 TPS (1.92×) | **55.80 TPS (1.92×)** |
| WMT-19 German–English translation | 3 | 29.80 TPS | 40.90 TPS (1.37×) | 42.50 TPS (1.43×) | **45.00 TPS (1.51×)** |

The largest supplied AdaEDL result is:

\[
56.90 / 25.74 \approx 2.21\times,
\]

on CNN-DM with maximum draft length 7. AdaEDL is not uniformly superior by a large margin: for example, on Dolly-15k with maximum draft length 16 it reaches 56.10 TPS versus 55.80 TPS for Max-Confidence-SPD.

### AdaEDL interpretation

AdaEDL changes the draft-generation loop rather than the target model. It observes draft-model entropy after generated tokens and stops when the entropy-derived acceptance criterion indicates insufficient expected benefit. The threshold is dynamically adapted using observed acceptance-rate statistics. The supplied facts do not provide exact numerical values for the controller quantities commonly denoted by parameters such as \(\alpha\), \(\epsilon\), \(\beta_1\), or \(\beta_2\); those values should not be inferred from this report.

The method still requires the separate Llama2-Drafter-115M model. Thus, “parameter-free” refers to the AdaEDL controller, not to the complete speculative-decoding system.

## 4. Verified Component-Aware Self-Speculative Results

The paper evaluates WikiText-2 validation with greedy decoding and draft lengths \(k\in\{2,4,8\}\). The supplied facts report all-token acceptance rates with bootstrap confidence intervals.

| Model / configuration | k=2 | k=4 | k=8 |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.038 [0.018, 0.060] | 0.019 [0.009, 0.030] | 0.009 [0.004, 0.015] |
| Falcon-H1-0.5B | 0.680 [0.620, 0.738] | 0.370 [0.335, 0.405] | 0.186 [0.168, 0.203] |
| Qwen2.5-0.5B with LayerSkip control | 0.520 [0.468, 0.573] | 0.326 [0.285, 0.370] | 0.179 [0.153, 0.206] |
| Falcon-H1-3B | 0.590 [0.528, 0.650] | 0.351 [0.310, 0.395] | 0.186 [0.161, 0.212] |

### Agreement and task acceptance

| Model / configuration | Top-1 agreement | Mean total-variation distance | Task acceptance at k=4, T=0 |
|---|---:|---:|---|
| Qwen3.5-0.8B | 0.203 | 0.803 | MMLU 0.013; GSM8K 0.001; Alpaca 0.011 |
| Falcon-H1-0.5B | 0.658 | 0.302 | MMLU 0.208; GSM8K 0.495; Alpaca 0.300 |
| Qwen2.5-0.5B with LayerSkip control | 0.496 | 0.473 | MMLU 0.268; GSM8K 0.106; Alpaca 0.126 |
| Falcon-H1-3B | 0.671 | 0.307 | Not reported in supplied data |

### Attention-ablation predictor

| Model | Baseline PPL | No-attention PPL | Ratio | Acceptance at k=4 |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 7.624 | 624.843 | **81.96×** | 0.019 |
| Falcon-H1-0.5B | 5.621 | 17.725 | **3.15×** | 0.370 |

These results support the paper’s central architectural observation: parallel hybrid compositions such as Falcon-H1 can yield useful component agreement, whereas sequential compositions such as Qwen3.5 can yield very poor acceptance. However, acceptance alone does not establish a serving-speed improvement.

## 5. Memory, Hardware, and Systems Caveats

- **AdaEDL:** The supplied evaluation uses one NVIDIA A100 80 GB in FP32. It retains the separate drafter and therefore does not eliminate drafter weights or draft-side execution state. No AdaEDL-specific VRAM comparison is supplied.
- **Component-aware self-speculation:** The method uses no separate drafter parameters, but the supplied facts report no VRAM measurement, hardware specification, end-to-end latency, TPS, energy result, or exact KV-cache-management implementation. Lower parameter overhead should therefore not be equated with a measured memory reduction or speedup.
- Neither supplied paper is reported to introduce a new KV-cache compression or sharing algorithm.
- Practical speed depends on draft execution cost, target verification cost, state/cache handling, kernels, batching, and scheduling overhead.

## 6. Artifacts and Reproduction

### AdaEDL

- Paper: [https://arxiv.org/abs/2410.18351](https://arxiv.org/abs/2410.18351)
- Target model: Llama2-7B
- Draft model: Llama2-Drafter-115M
- Datasets: CNN-DM, Dolly-15k, and WMT-19 German–English translation
- Hardware and precision: one NVIDIA A100 80 GB, FP32
- Code repository: **No repository URL provided in the supplied facts**
- Checkpoint URL: **None provided in the supplied facts**

### Component-Aware Self-Speculative Decoding

- Paper: [https://arxiv.org/abs/2605.01106](https://arxiv.org/abs/2605.01106)
- Official code: [https://github.com/hecboar/hybrid-speculative-decoding](https://github.com/hecboar/hybrid-speculative-decoding)
- Models: Qwen3.5-0.8B, Falcon-H1-0.5B, Falcon-H1-3B, and Qwen2.5-0.5B with LayerSkip control
- Dataset: WikiText-2 validation
- Reported protocol details: evaluated variants, prompt count, context truncation, draft lengths, temperatures, and bootstrap procedure are specified in the supplied paper facts
- Hardware: not reported in the supplied facts
- Checkpoint URL: **None provided in the supplied facts**

## Overall Assessment

AdaEDL is the more directly validated deployment optimization in this comparison because it reports end-to-end TPS on multiple workloads and maximum draft lengths. Its strongest supplied result is **2.21× over autoregressive decoding** on CNN-DM.

Component-aware self-speculation is a promising architecture-native approach that avoids a separately parameterized drafter. Its reported acceptance results are encouraging for Falcon-H1 but extremely weak for Qwen3.5. Because the paper does not supply end-to-end serving measurements or VRAM figures in the supplied facts, its production speed and memory benefits remain to be demonstrated experimentally.

The defensible practical conclusion is therefore:

1. AdaEDL has measured throughput evidence on the reported Llama2 target/drafter setup.
2. Component-aware self-speculation may reduce separate-drafter parameter overhead, but its usefulness is architecture-dependent.
3. Acceptance rates must not be converted into speedups without wall-clock draft, verification, cache, and scheduling measurements.