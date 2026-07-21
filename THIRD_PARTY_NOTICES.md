# Third-Party Notices

MSK-Bench includes adapted baseline implementations, vendored compatibility code, and bundled musculoskeletal model assets from third-party projects. Preserve every upstream license file in its original directory when redistributing this repository or derived packages.

This notice is a packaging aid for maintainers and downstream users. It is not legal advice, and it does not replace review of the individual license files.

## Component Index

| Component | Local path | Upstream | License files | Local MSK-Bench changes |
|---|---|---|---|---|
| depRL baseline | `depRL/` | https://github.com/martius-lab/depRL | `depRL/LICENSE` (Apache-2.0), `depRL/LICENSE.depRL` (MIT) | Local MSK-Bench changes add benchmark task wiring, evaluator entry points, artifact arguments, result export, and compatibility glue for MSK-Bench environments. |
| DynSyn/msgym baseline | `msgym/` | https://github.com/Beanpow/DynSyn | `msgym/LICENSE` (Apache-2.0) | Local MSK-Bench changes add benchmark task wiring, Stable-Baselines3 evaluator wrappers, artifact arguments, result export, and compatibility glue for MSK-Bench environments. |
| depRL middleware package | `deprl_middleware_22tasks/` | MSK-Bench local integration built around depRL-style latent-action workflows | See `depRL/LICENSE`, `depRL/LICENSE.depRL`, and the top-level license status. | Local MSK-Bench changes provide 22-task middleware configs, expert-data collection, transformer training, latent-action wrappers, and evaluator entry points. |
| MuscleMimic integration | `third_party/musclemimic/` | https://github.com/amathislab/musclemimic | `third_party/musclemimic/LICENSE` (Apache-2.0) | Local MSK-Bench changes adapt the upstream package for local benchmark imports, ResidualRL-related interoperability, task adapters, and environment integration. |
| tonic | `depRL/deprl/vendor/tonic/` | https://github.com/fabiopardo/tonic | `depRL/deprl/vendor/tonic/LICENSE` (MIT) | Vendored dependency retained inside the depRL tree. Preserve the vendored notice and license text. |
| MS-Human-700 model assets | `msk_bench/simhive/ms_human_700/` | MS-Human-700 model asset bundle | `msk_bench/simhive/ms_human_700/LICENSE` (Apache-2.0) | Bundled model assets are used by the local simulation environment path integration. |

## Maintainer Notes

- Do not remove upstream license files when editing or packaging these directories.
- Do not replace the local baseline README files with upstream README files. The local README files document MSK-Bench-specific entry points and modifications.
- Record meaningful local changes to third-party code in `PATCHES.md` so downstream users can tell what differs from upstream releases.
- When syncing an upstream project, update this file, the affected local README, and `PATCHES.md` in the same change.
