# Multi-Agent Systems for Autonomous Scientific Discovery and Adaptive Coordination: A Verified Comparative Analysis of MASTER and AOAD-MAT

## 1. Executive Summary

The two papers address different multi-agent settings:

1. **MASTER** — *Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery* — combines LLM-based scientific hypothesis formation and multi-agent deliberation with autonomous DFT workflow generation, execution, validation, and feedback.
2. **AOAD-MAT** — *AOAD-MAT: Transformer-based multi-agent deep reinforcement learning model considering agents' order of action decisions* — augments the Multi-Agent Transformer with an autoregressive subtask that predicts both an agent's action and which agent should act next.

They are not directly comparable on conventional systems metrics. The supplied ground truth reports no verified AOAD-MAT numerical benchmark results, memory figures, latency measurements, or hardware specifications. MASTER reports a reduction in the number of required atomistic simulations, not an equivalent wall-clock speedup.

## 2. Contributions and Coordination Mechanisms

### MASTER

MASTER's central modification over procedural automation is a hierarchical reasoning loop. Design agents use prior first-principles results and chemically motivated reasoning to select the next material, while specialized simulation and review agents execute and verify calculations. The workflow includes:

- scientific hypothesis formation and candidate selection;
- multi-agent deliberation;
- geometry and DFT-workflow generation;
- execution and structural/file validation;
- scientific review of computed results; and
- feedback of results into subsequent candidate selection.

The supplied implementation description specifies OpenAI Agents SDK v0.0.18, Codex v0.57.0, ASE, VASP POSCAR generation, isolated temporary execution directories, version-controlled rejected structures, and structured Markdown logs containing agent decisions, prompts, and generated scripts. It also describes geometry refinement with a maximum of five cycles.

### AOAD-MAT

AOAD-MAT augments MAT with autoregressive agent-order prediction. At each decoding stage, the policy jointly predicts:

- the current agent's action; and
- which agent should act next.

The predicted order dynamically permutes decoder latent observation representations. Action and ordering predictions are optimized through a shared PPO objective using the product of their policy ratios. Conceptually, if the action and order ratios are respectively $r_m^a$ and $r_m^i$, the joint ratio is:

$$
r_m = r_m^a r_m^i.
$$

The supplied material does not provide verified numerical results for the reported SMAC or MA-MuJoCo tasks.

## 3. Verified Benchmark Matrix

| Method | Benchmark / Dataset | Verified result | Memory / hardware / code status |
|---|---|---|---|
| **MASTER** | CO adsorption on transition-metal adatoms supported on Cu(100) | Up to **90% reduction in required atomistic simulations** relative to trial-and-error selection | Memory, VRAM, latency, and hardware not reported |
| **MASTER** | CO adsorption on M–N–C single-atom catalysts | Up to **90% reduction in required atomistic simulations** relative to trial-and-error selection | Memory, VRAM, latency, and hardware not reported |
| **MASTER** | CO adsorption-energy search across **28 transition metals from Sc to Au** | **N/A** for a separate numerical speedup or simulation-reduction figure | DFT settings and complete experimental protocol not supplied |
| **AOAD-MAT** | SMAC — 5m_vs_6m | **N/A** | No verified numerical result, memory figure, hardware specification, checkpoint, or complete implementation release supplied |
| **AOAD-MAT** | SMAC — MMM2 | **N/A** | Same limitations |
| **AOAD-MAT** | SMAC — 6h_vs_8z | **N/A** | Same limitations |
| **AOAD-MAT** | SMAC — 3s5z_vs_3s6z | **N/A** | Same limitations |
| **AOAD-MAT** | MA-MuJoCo — HalfCheetah (6×1) | **N/A** | Same limitations |

Repeated benchmark entries in the supplied extraction are likewise **N/A**; no additional verified AOAD-MAT metrics are inferred here.

## 4. Interpretation of Metrics

MASTER's reported 90% figure is a reduction in **required atomistic simulations** relative to trial-and-error selection. It must not be restated as:

- a 90% wall-clock speedup;
- a 90% reduction in total compute cost;
- a 90% reduction in energy consumption; or
- a measured GPU/VRAM efficiency improvement.

The supplied data provide no such measurements.

For AOAD-MAT, the available ground truth establishes the architectural contribution but supplies no numerical benchmark outcomes. Consequently, this report makes no claim that AOAD-MAT improves win rate, reward, training speed, or performance over MAT on any listed task.

Neither paper supplies verified measurements for:

- end-to-end latency;
- throughput;
- CPU/GPU utilization;
- VRAM or memory footprint;
- parameter count;
- energy consumption;
- cost per successful outcome; or
- scaling with agent count.

## 5. Architectural Comparison

| Dimension | MASTER | AOAD-MAT |
|---|---|---|
| Domain | Autonomous functional-materials discovery | Cooperative multi-agent reinforcement learning |
| Agent abstraction | LLM reasoning, simulation, and review/tool agents | Neural policy components representing cooperating agents |
| Coordination | Hierarchical deliberation and scientific feedback | Learned autoregressive action-order prediction |
| Feedback | First-principles results, structural checks, and reviewer feedback | PPO advantage signal from environment interaction |
| Main objective | Reduce expensive atomistic simulations while guiding discovery | Jointly optimize action selection and agent ordering |
| Runtime verification | Structural and workflow validation, including review and retries | No separate runtime artifact-validation loop is established in the supplied facts |
| Reported quantitative evidence | Up to 90% fewer required atomistic simulations on two materials tasks | No verified numerical benchmark metrics supplied |

MASTER is best characterized as an autonomous scientific workflow architecture. AOAD-MAT is best characterized as an order-aware MARL policy architecture. The papers should not be presented as competing systems on a common speed or memory benchmark.

## 6. Reproducibility and Artifacts

### MASTER

- Paper: [arXiv:2512.13930](https://arxiv.org/abs/2512.13930)
- Direct PDF: [https://arxiv.org/pdf/2512.13930](https://arxiv.org/pdf/2512.13930)
- Official code URL identified in the supplied facts: [https://github.com/openai/openai-agentspython](https://github.com/openai/openai-agentspython)

The supplied text specifies OpenAI Agents SDK v0.0.18, Codex v0.57.0, ASE, VASP POSCAR generation, isolated temporary execution directories, version-controlled rejected structures, and structured Markdown logs. It does **not** provide a dedicated MASTER code repository, model checkpoint, exact prompt package beyond supplementary excerpts, DFT settings, hardware specification, or complete experimental protocol.

### AOAD-MAT

- Paper: [arXiv:2510.13343](https://arxiv.org/abs/2510.13343)
- Direct PDF: [https://arxiv.org/pdf/2510.13343](https://arxiv.org/pdf/2510.13343)

The supplied text mentions MAT references and supplementary material containing the training algorithm and preliminary product-versus-weighted-loss evaluations. It does **not** provide a verified AOAD-MAT code repository, checkpoint, exact hyperparameter table, hardware specification, or complete implementation release.

## 7. Overall Assessment

MASTER and AOAD-MAT represent complementary approaches:

- **MASTER** coordinates heterogeneous reasoning, simulation, validation, and scientific-review agents in a closed loop. Its verified quantitative result is up to 90% fewer required atomistic simulations on two specified CO-adsorption discovery tasks.
- **AOAD-MAT** makes agent ordering an explicit autoregressive policy variable and couples ordering with action selection through a product-ratio PPO objective. Its benchmark metrics are not numerically available in the supplied ground truth and are therefore reported as N/A.

The evidence supports architectural comparisons, but not claims of generic multi-agent superiority, lower latency, lower memory use, or greater production efficiency. Further evaluation would require complete code and configuration artifacts, hardware details, reproducible protocols, and direct measurements of cost, latency, memory, failure rates, and scaling.