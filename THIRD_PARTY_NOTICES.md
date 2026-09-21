# Third-Party Notices

MSK-Bench includes adapted baseline implementations, vendored compatibility code, and bundled musculoskeletal model assets from third-party projects. Preserve every upstream license file in its original directory when redistributing this repository or derived packages.

## Additional source-release attribution

- LocoMuJoCo compatibility sources inside `rl_paradigms/musclemimic/loco_mujoco/` originate from [LocoMuJoCo](https://github.com/robfiras/loco-mujoco). The upstream [MIT notice](https://github.com/robfiras/loco-mujoco/blob/master/LICENSE) is supplied at `licenses/loco_mujoco-MIT.txt`; the embedded source tree remains unchanged.
- RoboHive-derived environment/robot/utilities code retains its original source and Apache-2.0 headers, including attribution to [RoboHive](https://github.com/vikashplus/robohive).
- `msk_bench/simhive/msk_sim/` retains embedded model attribution. Do not infer model/data redistribution rights solely from Python code licenses.
- See [data and license notes](docs/data-and-licenses.md) for retained author-generated motions, excluded restricted motions, network checkpoints, and asset-license boundaries.

This notice is a packaging aid for maintainers and downstream users. It is not legal advice, and it does not replace review of the individual license files.

## Component Index

| Component | Local path | Upstream | License files | Local MSK-Bench changes |
|---|---|---|---|---|
| depRL baseline | `rl_paradigms/depRL/` | https://github.com/martius-lab/depRL | `rl_paradigms/depRL/LICENSE` (Apache-2.0), `rl_paradigms/depRL/LICENSE.depRL` (MIT) | Local MSK-Bench changes add benchmark task wiring, evaluator entry points, artifact arguments, result export, and compatibility glue for MSK-Bench environments. |
| DynSyn/msgym baseline | `rl_paradigms/msgym/` | https://github.com/Beanpow/DynSyn and https://github.com/LNSGroup/msgym | `rl_paradigms/msgym/LICENSE` (Apache-2.0 from the msgym distribution) | Local MSK-Bench changes add benchmark task wiring, Stable-Baselines3 evaluator wrappers, artifact arguments, result export, and compatibility glue for MSK-Bench environments. |
| depRL middleware package | `rl_paradigms/deprl_middleware_22tasks/` | MSK-Bench local integration built around depRL-style latent-action workflows | Original wrapper contributions use Apache-2.0; depRL-derived portions retain `rl_paradigms/depRL/LICENSE` and `rl_paradigms/depRL/LICENSE.depRL`. | Local MSK-Bench changes provide 22-task middleware configs, expert-data collection, transformer training, latent-action wrappers, and evaluator entry points. |
| MuscleMimic integration | `rl_paradigms/musclemimic/` | https://github.com/amathislab/musclemimic | `rl_paradigms/musclemimic/LICENSE` (Apache-2.0) | Local MSK-Bench changes adapt the upstream package for local benchmark imports, ResidualRL-related interoperability, task adapters, and environment integration. |
| tonic | `rl_paradigms/depRL/deprl/vendor/tonic/` | https://github.com/fabiopardo/tonic | `rl_paradigms/depRL/deprl/vendor/tonic/LICENSE` (MIT) | Vendored dependency retained inside the depRL tree. Preserve the vendored notice and license text. |
| MS-Human-700 model assets | `msk_bench/simhive/ms_human_700/` | MS-Human-700 model asset bundle | `msk_bench/simhive/ms_human_700/LICENSE` (Apache-2.0) | Bundled model assets are used by the local simulation environment path integration. |

## Maintainer Notes

- Do not remove upstream license files when editing or packaging these directories.
- Do not replace the local baseline README files with upstream README files. The local README files document MSK-Bench-specific entry points and modifications.
- Record meaningful local changes to third-party code in `PATCHES.md` so downstream users can tell what differs from upstream releases.
- When syncing an upstream project, update this file, the affected local README, and `PATCHES.md` in the same change.

## Functional acceptance repairs

The depRL/tonic and DynSyn/msgym adapters also include short-training overrides, checkpoint reload checks and exceptional-exit resource cleanup. Middleware adds matching configuration and rendering compatibility. See `PATCHES.md` for details. The MuscleMimic source and its official network-pretrained checkpoint integration were not changed by this functional-acceptance repair.
