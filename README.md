# Benchmark-as-Teacher

[English](README.md) | [简体中文](README_zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2608.16211-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.16211)
[![Blog](https://img.shields.io/badge/Blog-Benchmark--as--Teacher-76B900?logo=google-chrome&logoColor=white)](https://kumakuma2002.github.io/bat/)
[![Hugging Face Data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Training%20Data-yellow)](https://huggingface.co/collections/MitakaKuma/benchmark-as-teacher)
[![AutoMedBench Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20AutoMedBench-Leaderboard-green)](https://huggingface.co/spaces/MitakaKuma/AutoMedBench-Leaderboard)

Benchmark-as-Teacher (BaT) is a self-evolving post-training framework that turns a structured benchmark into training infrastructure for long-horizon agents. BaT closes the loop between evaluation and training: stage-level rubrics diagnose weaknesses, a held-out-safe data bank supplies targeted practice, and grouped rollouts are optimized with GRPO. The updated checkpoint is re-evaluated by the same benchmark contract, creating a recursive improvement cycle.

![BaT Method](docs/assets/fig_method.png)

## Quick Start

See [docs/process.md](docs/process.md) for the full training and loop process.

```bash
# 1. Build the held-out-safe source grid
python -m scripts.build_bat_semantic_source --output-dir /path/to/safe-source

# 2. One-time cold-start SFT
python -m scripts.prepare_sft_dataset --source /path/to/safe-source/source.jsonl --output-dir /path/to/sft-data
python -m scripts.sft_train --data /path/to/sft-data --output-dir /path/to/sft-checkpoint

# 3. Build a BaT round
python -m scripts.build_bat_core_dataset \
  --source /path/to/safe-source/source.jsonl \
  --source-sha256 "$(sha256sum /path/to/safe-source/source.jsonl | cut -d' ' -f1)" \
  --round-id r01 --target-stage S3 \
  --output-dir /path/to/r01-data

# 4. Launch GRPO round
bash rl/grpo/launch_core_grpo.sh

# 5. Evaluate and gate
python -m scripts.run_bat_core_eval --manifest /path/to/r01-data/manifest.json --output-dir /path/to/r01-eval
python -m scripts.check_bat_core_gate --baseline /path/to/baseline-eval --candidate /path/to/r01-eval --output-dir /path/to/r01-gate

# 6. Run the full loop
python -m scripts.run_bat_core_loop --rounds 10 --output-dir /path/to/bat-run
```

## BaT

### Stage Bank

Stage Bank is the asynchronous data pipeline that synthesizes held-out-safe training states outside the policy-update loop. It fabricates fictional tasks, reconstructs executable stage states, and applies leakage checks before any data reaches training.

The reference grid contains one abstract state for every `focus x track x outcome` cell:

```text
6 focus values (E2E and S1-S5)
x 7 benchmark tracks
x 3 outcomes (success, recovery, failure)
= 126 held-out-safe source states
```

Each state is content-isolated: task identities, answers, reports, traces, and correction text never cross the evaluation firewall.

### BiCuRL

Bilevel Curriculum Reinforcement Learning (BiCuRL) is the self-improving post-training method. Its outer loop reads stage scores from a fixed held-out evaluation, selects the next target stage, and retains or rejects candidate checkpoints. Its inner loop samples Stage Bank states, scores new rollouts with rubric items and artifact evidence, and updates the policy with GRPO.

## BaT Agent

A trained policy becomes a BaT Agent when paired with a fixed execution environment that includes public stage skills. The engineering layer is kept fixed across comparisons so BiCuRL changes only the policy.

![BaT Agent Performance](docs/assets/fig_teaser_figure.png)

On AutoMedBench-Lite, BaT-4B and BaT-9B more than double the Overall scores of their Qwen Instruct baselines. BaT-9B Agent reaches 79.6 Overall, exceeding Claude Opus 4.6 with Claude Code at 77.5.

## Infrastructure

BaT is designed as infrastructure. Training Engines and Inference Engines are independently swappable, and every combination below is supported:

| Training Engine | Inference Engine | Status |
|-----------------|------------------|--------|
| <img src="docs/assets/logos/verl.jpg" alt="VeRL" height="24"/> VeRL | <img src="docs/assets/logos/vllm.svg" alt="vLLM" height="24"/> vLLM | Supported |
| <img src="docs/assets/logos/verl.jpg" alt="VeRL" height="24"/> VeRL | <img src="docs/assets/logos/sglang.svg" alt="SGLang" height="24"/> SGLang | Supported |
| <img src="docs/assets/logos/slime.jpg" alt="Slime" height="24"/> Slime | <img src="docs/assets/logos/vllm.svg" alt="vLLM" height="24"/> vLLM | Supported |
| <img src="docs/assets/logos/slime.jpg" alt="Slime" height="24"/> Slime | <img src="docs/assets/logos/sglang.svg" alt="SGLang" height="24"/> SGLang | Supported |

## Citation

```bibtex
@article{liu2026bat,
  title={BaT: Towards Self-Evolving Medical Research Agent with Stage Rubrics},
  author={Liu, Junqi and He, Yufan and He, Yexiao and Guo, Pengfei and Yang, Dong and Myronenko, Andriy and Zhao, Can and Ye, Hanrong and Qi, Tianhao and Zhou, Yuyin and Xu, Daguang and Tang, Yucheng},
  year={2026},
  journal={arXiv preprint arXiv:2608.16211},
  url={https://arxiv.org/abs/2608.16211}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
