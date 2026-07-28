# Local Patch Summary

This file summarizes local MSK-Bench changes made to third-party or upstream-derived code. It is intended to help maintainers compare this repository with upstream projects before publishing releases, syncing new upstream commits, or redistributing modified baseline code.

## depRL/

Upstream: https://github.com/martius-lab/depRL

License references: `depRL/LICENSE` (Apache-2.0), `depRL/LICENSE.depRL` (MIT), and `depRL/deprl/vendor/tonic/LICENSE` (MIT for the vendored tonic copy).

Local MSK-Bench changes:

- Added evaluator scripts for success, robustness, smoothness, energy, EMG export, and rendering against MSK-Bench environments.
- Added MSK-Bench task and artifact argument compatibility, including benchmark-root, run-path, checkpoint, checkpoint-file, header, agent, environment, and env-expression style arguments.
- Added result export behavior for JSON/CSV outputs compatible with the shared benchmark schema.
- Kept vendored support code that should retain its upstream license and notice files.

## msgym/

Upstream: https://github.com/Beanpow/DynSyn

License reference: `msgym/LICENSE` (Apache-2.0).

Local MSK-Bench changes:

- Added evaluator scripts for success, robustness, smoothness, energy, EMG export, and rendering against MSK-Bench environments.
- Added Stable-Baselines3-style artifact loading flags for model roots, explicit model paths, normalization statistics, and logs.
- Added MSK-Bench task wiring and result export behavior for JSON/CSV outputs compatible with the shared benchmark schema.

## deprl_middleware_22tasks/

Upstream: this is a local MSK-Bench integration package that builds around depRL-style workflows rather than a direct upstream checkout.

License references: original MSK-Bench wrapper code is released under the top-level MIT License; depRL-derived components retain the depRL notices listed above.

Local MSK-Bench changes:

- Added latent-action middleware registration for the 22 canonical MSK-Bench tasks.
- Added configuration generation, expert-data collection, transformer training, and middleware evaluator scripts.
- Added explicit encoder/decoder artifact handling and strict-weight checks for reproducible middleware evaluation.

## third_party/musclemimic/

Upstream: https://github.com/amathislab/musclemimic

License reference: `third_party/musclemimic/LICENSE` (Apache-2.0).

Local MSK-Bench changes:

- Adapted local imports and package layout for MSK-Bench integration.
- Kept compatibility code needed by ResidualRL-style workflows and benchmark environment adapters.
- Added `third_party/musclemimic/README.MSK-Bench.md` so the local integration context is visible without overwriting upstream project documentation.

## Sync Checklist

When updating any upstream-derived directory:

- Preserve upstream license files and copyright notices.
- Keep local MSK-Bench README attribution sections intact.
- Re-run the regression tests, including `tests.test_third_party_attribution`.
- Update this file and `THIRD_PARTY_NOTICES.md` if paths, upstream URLs, licenses, or local MSK-Bench changes change.
