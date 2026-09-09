# Supplementary technical material

[Read the technical appendix (PDF)](supplementary-material.pdf)

The 24-page document preserves the technical appendix and its references from the current anonymous manuscript. Table and figure numbering are retained so it can be read alongside the main paper. The guide below highlights the material most relevant to interpreting and reproducing the reported comparisons.

## Reading guide

| Material | Location in the appendix | PDF pages |
| --- | --- | --- |
| Task descriptions, notation, rewards, observations, initialization, and termination | A; Tables VI-XI | 1-8 |
| Metric definitions and perturbation protocol | B | 8-9 |
| Full task-wise performance tables | Tables XII-XIV | 10-11 |
| Comparison scope and aggregation | C-D; Tables XV-XVI | 9-12 |
| Complete 22-task learning curves | F; Figure 11 | 13 |
| Reward tuning and latent-action representation | H-I | 13-17 |
| Human EMG processing and comparison | J | 17-18 |
| Anatomical grouping and load study | K-L | 18-19 |
| Residual, reward, and timing adaptation | M | 18-20 |
| Bibliographic sources | Appendix References | 23-24 |

## Benchmark scope

All 22 tasks use MuJoCo and the same 416-muscle full-body elastic-tendon embodiment. The task families are:

- **Stabilization (6):** stand, single-leg stand, squat, sit, balance, powerlift.
- **Locomotion (6):** walk, run, jump, crawl, sidestep, turn.
- **Interaction (10):** stairs, hurdles, stepping stones, sliding terrain, reach, catch, chin-up, door opening, pole walking, walk-and-sit.

PPO, SAC, DepRL, and DynSyn-SAC share the full 22-task reward-based comparison, with matched rewards and evaluation settings within each task. They use separately task-trained policies: this is not one universal policy or a zero-shot-transfer evaluation. The imitation robustness study covers stand, jump, walk, run, and stairs. Residual adaptation covers walk, run, and stairs. Reward tuning and action compression are focused studies, not five additional full-suite comparisons. The 700-muscle model belongs to the anatomy study, not the shared 22-task benchmark.

## Two distinct evaluation protocols

**Training-environment success (SR)** restores the complete environment used in training, including native noise. SR is the percentage of episodes satisfying task completion while remaining alive and within task-specific physical constraints. It is not a noiseless evaluation.

**Perturbation-sweep robustness** is reported separately. Gaussian perturbations affect normalized actions or observations; uniform perturbations change individual muscle force gains. Action and dynamics scales are 0, 0.05, 0.10, 0.15, 0.20; observation scales are 0, 0.02, 0.05, 0.08, 0.10. A zero sweep coordinate denotes zero on that axis and must not be substituted for the training-environment SR baseline.

Robustness area integrates success across a sweep using the trapezoidal rule, divides by that sweep's range, and averages equally over the 22 tasks and then the three perturbation types. Full-suite SR weights 22 tasks equally; family means weight tasks equally within that family. Nonzero coverage counts tasks with SR above zero, not reliably mastered tasks. Rankings describe point estimates, not statistical significance.

| Method | All-task SR (%) | Stabilization | Locomotion | Interaction | Mean robustness area | Nonzero SR coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DynSyn-SAC | 52.00 | 57.33 | 81.33 | 31.20 | 41.59 | 16/22 |
| DepRL | 48.36 | 50.33 | 41.33 | 51.40 | 43.67 | 22/22 |
| SAC | 12.82 | 42.33 | 0.00 | 2.80 | 11.37 | 4/22 |
| PPO | 3.36 | 0.00 | 0.00 | 7.40 | 3.14 | 1/22 |

| Method | Action area | Observation area | Dynamics area |
| --- | ---: | ---: | ---: |
| DynSyn-SAC | 46.66 | 43.37 | 34.75 |
| DepRL | 53.06 | 38.10 | 39.84 |
| SAC | 15.36 | 4.26 | 14.50 |
| PPO | 3.10 | 3.40 | 2.92 |

## Seven metric families

1. **Success rate:** task completion under the stated training-environment protocol.
2. **Robustness:** performance across action, observation, and dynamics sweeps.
3. **EMG-envelope similarity:** normalized muscle-waveform agreement where matched human recordings exist.
4. **Cumulative reward:** optimization on a task's own reward scale; raw returns are not pooled across tasks.
5. **Peak-efficiency steps:** training step of the highest evaluated return, not a measure of biological efficiency.
6. **Activation cost:** average squared activation across muscles and timesteps, an effort proxy rather than measured metabolic energy.
7. **Joint smoothness:** mean squared angular jerk, shown on a logarithmic scale. Jerk uses the second finite difference of angular velocity divided by the squared timestep.

Low activation cost or jerk can accompany failed behavior. For example, walking SAC has cost 0.1427 with 0% SR, whereas DepRL has cost 0.4206 with 64% SR. Quality metrics must therefore be read alongside task success and comparable behavior.

## EMG processing and interpretation

Profiles are resampled to 101 points per gait cycle using cubic interpolation, min-max normalized per muscle, and averaged over cycles. A cyclic shift is selected to maximize Pearson correlation with the human profile. The resulting score measures waveform shape after amplitude and phase alignment; it removes absolute amplitude and phase information.

Walking and stairs use Gait120 human recordings; running uses the 3 m/s Van Hooren-Meijer recordings. Bibliographic details are retained in the appendix. Running and stair means cover 12 matched muscles. The common walking table instead reports SoleusMed., Bic.Fem., and Semitendinosus separately; these different muscle sets must not be pooled into a cross-task EMG mean.

| Task | MuscleMimic SR (%) | Adapted SR (%) | MuscleMimic EMG r | Adapted EMG r |
| --- | ---: | ---: | --- | --- |
| Walking | 100 | 100 | 0.71 / 0.69 / 0.63 | 0.75 / 0.80 / 0.67 |
| Running | 90 | 100 | 0.43 | 0.77 |
| Stairs | 0 | 100 | 0.51 | 0.52 |

The extended 12-muscle comparison gives Residual / DepRL / MuscleMimic correlations of 0.77 / 0.70 / 0.43 for running and 0.52 / 0.56 / 0.51 for stairs. Thus, the adapted controller leads running but not stairs. Failed and successful policies may cover different portions of the behavior: these are not controlled comparisons of identical movements. A score of 0.51 for the failed stair baseline illustrates why waveform correlation alone cannot certify whole-task completion. No equivalence or statistical-independence claim is made.

## Combined adaptation, not a component-isolated method claim

A frozen MuscleMimic policy supplies the base action. A trainable three-layer MLP supplies residual outputs, clipped to [-1, 1], multiplied by residual authority alpha, added to the base action, and clipped to the environment action range [-1, 1].

The pipeline also changes task objectives and timing. Stairs relaxes vertical tracking and adjusts reference pacing; walking uses phase-dependent residual authority and activation regularization; running changes flight-phase rewards and penalizes ground-reaction forces. Standard kinematic penalties are disabled during adaptation, with a minimal residual L2 penalty of -0.02. Reported improvements concern this combined pipeline and do not identify the independent contribution of each change.

Stair height changes from 0.94 to 1.21 m and tracking error from 0.1968 to 0.1079 rad. Running tracking error changes from 0.2267 to 0.1338 rad and peak ground-reaction force from 6.40 to 6.07 times body weight. These diagnostics characterize the evaluated motion; they do not establish overall biological realism or prove the proposed stair timing explanation.

## Additional diagnostics

With reward weights constrained to norm 40, tuning improves tracking RMSE from 0.4446 to 0.4355 and reduces activation from 0.4769 to 0.3999, while survival falls from 417 to 246.6 steps. Better tracking and lower activation therefore do not alone demonstrate better control.

Compression from 416 to 128 action dimensions improves learning and several robustness results. From 128 to 16 dimensions, held-out reconstruction MSE rises from 0.0084 to 0.0188 and PCA explained variance falls from 89.4% to 65.4%. Loss of useful activation information is a plausible interpretation, not a uniquely established mechanism.

The load and anatomical-resolution studies describe simulated recruitment changes. They do not supply a matched human reference and therefore do not establish improved human similarity. Complete figures, anatomical grouping rules, and method equations are retained in the PDF.

## Release scope

This package contains project demonstrations and reported technical material, not code or checkpoints. It contains no new experiments beyond the manuscript. Selected rollout videos are illustrative; use the reported tables for aggregate performance. The main paper retains the evidence needed to support its conclusions independently of this supplement.
