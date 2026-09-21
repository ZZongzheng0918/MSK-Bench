# Functional validation record

## Metric-alignment validation (2026-09-20/21)

Windows, Python 3.12.7, in a dedicated validation virtual environment. Existing
third-party dependencies were reused; this was not a clean dependency installation.
An isolated interpreter confirmed that all six project package roots resolved to
the public-release worktree, not to another source checkout.

- Full regression run: **227 passed, 5 skipped, 0 failures**, 744.78 seconds.
  This covers 22 canonical environment create/reset/step checks, available
  extensions, representative short training/model reloads, and five baseline
  families' metric/export/render entry points. It is not full training on all
  algorithm/task combinations.
- After the last six focused cases were added, a fresh combined metric/analysis
  and release-contract run on 2026-09-21 passed **51 tests**, with one third-party
  scipy.misc deprecation warning. This includes all 29 metric/workflow cases.
  These totals overlap and must not be added together.
- Maintained-source Ruff checks passed. The wheel built successfully; inspection
  confirmed the new analysis modules and agentic extra, with no restricted walking
  references or generated GPT experiment results.
- The five full-suite skips were two missing authorized ResidualWalk/ResidualRun
  motion references and three opt-in pretrained-weight checks. None is reported
  as a pass. No paid LLM API call was made; agentic API tests use a simulated client.

Corrected formulas and remaining experimental prerequisites are documented in
[analysis workflows](analysis-workflows.md). Older metric outputs need recomputation.
Neither these tests nor the new analysis commands establish historical numerical
equivalence, control quality, or full reproduction of every paper experiment.

## Historical source-snapshot validation

Validation was performed on Windows with Python 3.11.9. All six project package roots (`msk_bench`, `deprl`, `msgym`, middleware, `musclemimic`, and `loco_mujoco`) resolved inside the repository in an isolated interpreter. Previously installed declared dependencies were reused; this was not a fresh network dependency installation. `pip check` and maintained-source lint passed.

| Check | Result and scope |
| --- | --- |
| Environment creation/reset/step | 22 canonical environments plus AgenticWalk and ResidualStair passed. Two gated-data environments skipped. |
| Real pretrained ResidualStair | Passed with cached official network-downloaded weights and a non-null base policy; not a zero-policy substitute. |
| Short training | Five baseline jobs passed on `MSKBenchSquat-v0`, eight steps each, saving validated model artifacts. |
| Evaluation | All 25 jobs passed: five baselines by five metric/export types. |
| Rendering | All five renderer jobs passed with decodable MP4 files. |
| Additional runtime regression | Covered model reloads, expert-data collection, one-epoch auxiliary Transformer training/reload, error handling, and resource cleanup. |
| Regression evidence | Full source suite: 205 passed, 5 skipped, 1 warning in 644.68 seconds. |
| Release checks | Layout, attribution, dependency, data-import, local-path, and generated-artifact checks passed. |

The five skips comprised two missing-motion environment cases and three opt-in pretrained cases. Pretrained Stairs was subsequently tested separately and passed; Walk/Run remain unavailable without authorized reference data. See the [README](../README.md#restricted-motion-references) for the unresolved exact Walk reference.

Retained author-generated motion arrays were byte-checked against the source used for validation. Heuristic secret and local-path scans found no matches; required third-party author notices remain. These are bounded checks, not proof of legal clearance or absence of every possible secret.

The initial public-release metadata conversion did not change runtime behavior.
Subsequent metric-alignment changes do change evaluation formulas, strength
perturbations, optional Powerlift mass, and agentic update constraints. See
[analysis workflows](analysis-workflows.md) for definitions and remaining
replication requirements. The historical results above are not fresh verification
of those changes. All smoke evidence is bounded: it does not establish convergence,
paper-result reproduction, every algorithm/environment combination, or
Linux/macOS runtime validation.
