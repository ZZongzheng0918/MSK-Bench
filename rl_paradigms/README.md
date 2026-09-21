# Reinforcement-Learning Paradigms

This directory groups the control and learning paradigms integrated with MSK-Bench. The main `msk_bench` package owns the environments and stable Gymnasium IDs; each directory here owns one training, evaluation, adaptation, or imitation approach.

| Directory | Paradigm |
|---|---|
| `agentic_walk/` | Agentic reward adaptation for walking. |
| `residualrl/` | Residual policies layered over a MuscleMimic base policy. |
| `ppo/` | Stable-Baselines3 PPO baseline. |
| `sac/` | Stable-Baselines3 SAC baseline. |
| `depRL/` | depRL/Tonic baseline and vendored compatibility code. |
| `msgym/` | DynSyn/msgym training and evaluation integration. |
| `deprl_middleware_22tasks/` | Latent-action middleware for all canonical tasks. |
| `musclemimic/` | MuscleMimic integration used by imitation and residual workflows. |

Run repository-level commands from the repository root. The independent nested projects keep their own metadata and licenses; install their optional dependencies only when using that paradigm. See the root `README.md`, `THIRD_PARTY_NOTICES.md`, and `PATCHES.md` for commands and attribution.
