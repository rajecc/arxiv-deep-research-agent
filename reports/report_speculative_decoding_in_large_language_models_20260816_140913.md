# Speculative Decoding in Large Language Models: Adaptive Draft-Length Control and CTC-Structured Parallel Drafting

## 1. Executive Summary

Speculative decoding accelerates autoregressive large-language-model inference by using a less expensive drafting mechanism to propose future tokens and then verifying them with the target model. The two papers considered here optimize different parts of this process:

1. **AdaEDL** — adaptive, training-free early stopping during sequential draft generation.
2. **CTC-drafter** — a trained, transformer-based draft module using sequence-level Connectionist Temporal Classification (CTC) and CTC-aware verification.

### High-level comparison

| Dimension | AdaEDL | CTC-drafter |
|---|---|---|
| Primary intervention | Adaptive stopping of sequential drafting | Improved parallel candidate generation and verification |
| Additional training | None | Required: draft module trained with CTC loss |
| Learned stopping predictor | None | Not applicable |
| Draft source | Llama2-Drafter-115M in the reported experiments | Transformer-based Attention Draft Module attached to Vicuna models |
| Main reported metric | Absolute throughput in tokens per second (TPS) | Relative speedup over vanilla autoregressive decoding |
| Best supplied result | 57.10 TPS on Dolly-15k with maximum draft length 7 | 2.78× on MT-bench |
| Memory reporting | Not reported | Not reported |

AdaEDL and CTC-drafter are conceptually complementary, but their combination was not evaluated in the supplied material.

---

## 2. Verified Benchmark Results

### 2.1 AdaEDL

The reported AdaEDL setup uses a Llama2-7B target model and Llama2-Drafter-115M draft model. The supplied facts identify a single NVIDIA A100 80 GB GPU and FP32 precision. Results are reported as tokens per second (TPS). “Base-SPD” is static speculative decoding, while “Max-Confidence-SPD” is the comparison training-free stopping method.

| Maximum draft length | Dataset | Autoregressive | Base-SPD | Max-Confidence-SPD | AdaEDL |
|---:|---|---:|---:|---:|---:|
| 16 | CNN-DM summarization | 25.74 | 36.30 | 49.50 | **54.10** |
| 16 | Dolly-15k creative writing | 29.02 | 32.10 | 55.80 | **56.10** |
| 16 | WMT-19 German–English translation | 29.80 | 22.30 | 43.70 | **43.90** |
| 7 | CNN-DM summarization | 25.74 | 51.50 | 53.50 | **56.90** |
| 7 | Dolly-15k creative writing | 29.02 | 47.60 | 56.60 | **57.10** |
| 7 | WMT-19 German–English translation | 29.80 | 32.70 | 45.10 | **45.20** |
| 3 | CNN-DM summarization | 25.74 | 54.10 | 53.10 | **55.70** |
| 3 | Dolly-15k creative writing | 29.02 | 54.70 | 55.70 | **55.80** |
| 3 | WMT-19 German–English translation | 29.80 | 40.90 | 42.50 | **45.00** |

The abstract reports improvements of **10%–57% over static speculative decoding** and improvements of **up to 10% over other training-free draft-stopping methods**. These percentages should be understood as the paper’s reported aggregate claims; the detailed table above reports absolute TPS.

### 2.2 CTC-drafter

The supplied material identifies evaluations on Vicuna-7B, Vicuna-13B, and Vicuna-33B using MT-bench and GSM8K. The following values are the supplied speedup results relative to vanilla autoregressive decoding.

| Method | Model | Benchmark | Speedup |
|---|---|---|---:|
| CTC-drafter | Vicuna-7B | MT-bench | **2.78×** |
| CTC-drafter | Vicuna-13B | MT-bench | **2.52×** |
| CTC-drafter | Vicuna-33B | MT-bench | **2.20×** |
| CTC-drafter | Vicuna-7B | GSM8K | **2.43×** |
| CTC-drafter | Vicuna-13B | GSM8K | **2.66×** |
| CTC-drafter | Vicuna-33B | GSM8K | **2.16×** |
| Medusa baseline | Vicuna-7B | MT-bench | 2.13× |
| Medusa baseline | Vicuna-13B | MT-bench | 1.97× |
| Medusa baseline | Vicuna-33B | MT-bench | 1.93× |
| Medusa baseline | Vicuna-7B | GSM8K | 2.33× |
| Medusa baseline | Vicuna-13B | GSM8K | 2.21× |
| Medusa baseline | Vicuna-33B | GSM8K | 2.10× |
| Hydra baseline | Vicuna-7B | MT-bench | 2.36× |
| Hydra baseline | Vicuna-13B | MT-bench | 2.17× |
| Hydra baseline | Vicuna-33B | MT-bench | 2.15× |
| Vanilla autoregressive baseline | Evaluated Vicuna models | MT-bench and GSM8K | 1.00× |

#### CTC-drafter ablation

| Configuration | Benchmark | Speedup |
|---|---|---:|
| Transformer draft + CTC loss + CTC verification | MT-bench | **2.78×** |
| Transformer draft + CTC loss + Medusa verification | MT-bench | **2.25×** |

The supplied facts do **not** provide hardware, precision, batch size, context length, memory usage, complete training configuration, or runtime-overhead percentages for CTC-drafter.

---

## 3. Methodological Verification

### 3.1 AdaEDL

AdaEDL adaptively stops draft generation using an entropy-based approximation to a lower bound on the expected acceptance probability of the next draft token. When the observed draft-logit entropy indicates that another token is unlikely to be accepted, the method stops drafting rather than continuing to a static maximum length.

The method is reported as:

- training-free;
- free of a learned draft-stopping predictor;
- applicable to conventional sequential speculative decoding;
- dependent on an existing draft model.

The supplied excerpt refers to Algorithm 1 and Section 4.4 for threshold-update details, but does not include the exact entropy-bound equation, complete threshold-update specification, or complete hyperparameter values. Those details should not be reconstructed from the excerpt.

### 3.2 CTC-drafter

CTC-drafter proposes a draft module trained with sequence-level CTC loss. Unlike independent linear prediction heads, it uses a transformer layer to model relationships among draft positions. CTC handles blank symbols and repeated tokens, allowing candidate sequences to be collapsed before verification.

At verification time, CTC-collapsed candidate sequences are used and the target model’s attention mask is modified so that positions removed by the collapse operation are ignored. The intended effect is higher-quality candidates and higher acceptance rates.

The supplied material does not include the paper’s exact CTC equation, candidate-generation details, attention-mask formula, or complete implementation specification. Such formulas should therefore be cited only from the full paper, not presented as reconstructed paper equations.

---

## 4. Engineering Interpretation

The papers report different metrics and should not be treated as a unified leaderboard:

- AdaEDL reports absolute TPS on a specified A100 80 GB, FP32 setup.
- CTC-drafter reports relative speedups, while the supplied facts do not identify its hardware or precision.
- AdaEDL uses a separate 115M draft model in the reported experiment.
- CTC-drafter uses a trained transformer-based draft module attached to Vicuna models.
- Neither supplied excerpt reports complete VRAM measurements.

### Practical trade-offs

| Criterion | AdaEDL | CTC-drafter |
|---|---|---|
| Training requirement | None for the stopping mechanism | Draft module training required |
| Integration | Lower complexity when conventional speculative decoding is already available | Higher complexity because of the custom draft module and CTC-aware verification |
| Runtime mechanism | Entropy calculation and adaptive stopping | Parallel draft computation, CTC collapse, and modified verification |
| Main risk | Entropy may imperfectly predict target acceptance | Training and implementation may be model- or deployment-specific |
| Memory evidence | No explicit VRAM result supplied | No explicit VRAM result supplied |
| Immediate deployment posture | More readily deployable in an existing speculative-decoding system | More specialized and training-dependent |

The supplied evidence does not establish a definitive memory ranking, latency distribution, batch-scaling behavior, hardware portability result, or production advantage for either method.

---

## 5. Reproducibility and Artifacts

### AdaEDL

- **Paper:** [arXiv:2410.18351](https://arxiv.org/abs/2410.18351)
- **PDF:** [https://arxiv.org/pdf/2410.18351](https://arxiv.org/pdf/2410.18351)
- **Code repository:** No GitHub repository was included in the supplied facts.
- **Checkpoint URL:** None was included in the supplied facts.

The supplied AdaEDL information identifies the datasets, target and draft models, GPU, precision, and maximum draft lengths. Reproduction remains incomplete because the supplied excerpt does not include the exact entropy-bound equation, complete threshold hyperparameters, or implementation artifacts.

### CTC-drafter

- **Paper:** [arXiv:2412.00061](https://arxiv.org/abs/2412.00061)
- **PDF:** [https://arxiv.org/pdf/2412.00061](https://arxiv.org/pdf/2412.00061)
- **Code repository:** No GitHub repository was included in the supplied facts.
- **Checkpoint URL:** None was included in the supplied facts.

The supplied CTC-drafter information establishes the principal architecture, evaluated Vicuna model sizes, benchmarks, speedups, and ablation values. It does not provide hardware specifications, hyperparameters, training configuration, checkpoint links, or source code.

---

## Overall Technical Assessment

AdaEDL is a runtime draft-length control method that reduces wasted sequential drafting without additional training or a learned stopping predictor. Its supplied results show that adaptive stopping can outperform static speculative decoding across CNN-DM, Dolly-15k, and WMT-19, including cases where static speculative decoding is slower than autoregressive decoding.

CTC-drafter is a trained draft-generation and verification architecture. Its CTC objective and CTC-aware candidate verification are intended to improve candidate structure and acceptance. The supplied results report speedups from **2.16× to 2.78×**, with the best value of **2.78×** on MT-bench.

The strongest supported conclusion is that the methods address complementary bottlenecks: AdaEDL controls **how long** to draft, while CTC-drafter changes **how candidates are generated and verified**. The supplied papers do not evaluate their combination, and they do not provide enough memory, hardware, or deployment data to support stronger production-level claims.