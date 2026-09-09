# MSK-Bench: Anonymous Research Materials

Benchmarking full-body musculoskeletal motor control across tasks, learning paradigms, and physiological metrics.

This review package contains the existing project demonstrations and the supplementary technical material. Author identities and personal-homepage links are omitted from this version.

- [Project page source](index.html): the original project layout, task galleries, and updated findings.
- [Protocols, findings, and appendix reading guide](materials/README.md).
- [Technical appendix (PDF, 24 pages)](materials/supplementary-material.pdf): task definitions, reward functions, observations, termination criteria, metrics, complete result tables, and method details.
- [Task demonstrations](static/videos/tasks/).
- [Policy-by-task video gallery](static/videos/gallery/).

The shared benchmark uses **22 tasks, one 416-muscle embodiment, and seven metric families**. Four reward-based algorithms share the 22-task comparison; focused studies span five paradigms. The 700-muscle model is used only in the anatomy study.

## What the results show

1. **Average success and coverage favor different methods.** DynSyn-SAC has 52.00% mean SR with nonzero success on 16/22 tasks; DepRL has 48.36% with nonzero success on 22/22. Coverage does not mean reliable mastery.
2. **Stairs is a diagnostic boundary of the evaluated motion prior.** The dynamics-perturbation comparison favors MuscleMimic on stand, jump, walk, and run, but DepRL on stairs. Tracking errors document divergence; its cause is not established.
3. **The largest success gain is not the largest EMG gain.** Stair adaptation changes SR from 0% to 100% and correlation from 0.51 to 0.52; running changes SR from 90% to 100% and correlation from 0.43 to 0.77.

Training-environment SR retains native training noise. Separate perturbation sweeps measure robustness. EMG scores compare normalized, phase-aligned waveform shape, not overall biological realism.

The main paper contains the evidence required for its conclusions. This package supplies additional detail, not new experiments. Code and checkpoints are not included in this release. Videos are selected demonstrations, not a replacement for aggregate evaluation.
