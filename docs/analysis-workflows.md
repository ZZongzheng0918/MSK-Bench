# Quantitative analysis workflows

These commands consume actual user-supplied evaluation artifacts. Generated data,
plots, checkpoints and human reference data are not bundled. A working entry point
does not demonstrate convergence or reproduce a published numerical result.

## Corrected metric definitions

- Peak efficiency is the earliest **evaluated** environment step attaining maximum
  mean return, not the last logged step. Repeated seed means at a step receive equal
  weight. Supply a consistent seed set and evaluation schedule; missing/nonfinite
  step-return pairs are omitted. Returns from different tasks are never pooled.
- Joint jerk is the second finite difference of joint angular velocity divided by
  dt squared; its mean square is reported as joint_jerk_energy. Ball and hinge DOFs
  are included, free-root and sliding DOFs excluded. The log10 field uses a floor of
  1e-12. At least three velocity samples are required; shorter rollouts fail rather
  than reporting a misleading zero. Action jerk remains the third difference of
  the action signal. Shared rollout collectors sample the pre-step physical state
  so vector-environment auto-reset cannot contaminate terminal samples.
- Robustness uses trapezoidal integration divided by the tested scale range, on
  a 0--100 scale, followed by equal weighting over tasks and perturbation types.
  Action and strength scales are 0, .05, .10, .15, .20; observation scales are
  0, .02, .05, .08, .10. Each task must have every type and matching grids.
- For maintained baseline evaluators, dynamics noise perturbs muscle strength,
  not muscle length range: MuJoCo force parameter 2, or scale parameter 3 when
  force is inferred automatically. Active and passive parameters are both scaled.
  See the [MuJoCo muscle specification](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-muscle).
  Apply once to a fresh model per sweep point. The vendored MuscleMimic evaluator
  is unchanged and must not be assumed to use this corrected protocol.

Old smoothness, peak-efficiency and dynamics-robustness outputs must be recomputed
before comparison with corrected outputs. These changes do not retroactively
validate historical tables.

## Robustness and training efficiency

Run the existing robustness evaluators, then collect one algorithm's rows in one
CSV or JSON list. Required columns: algorithm, env_id, noise_type (action, obs,
dynamics), noise_scale, success_rate. Evaluator success_rate values are percentages.
Repeated rows at a scale are averaged equally (use one mean per seed).

```bash
python -B -m benchmark_eval.analyze robustness --input results/robustness_ppo.csv --success-unit percent --all-tasks --output results/robustness_auc.json
python -B -m benchmark_eval.analyze peak --input results/evaluation_history.csv --output results/peak_steps.json
```

The first command rejects missing canonical tasks and non-paper grids. For a
deliberate task subset omit --all-tasks; for a different grid use --custom-grid and
do not label its score as the standard benchmark score. Units are never inferred.
Peak input requires algorithm, env_id, step and mean_return. Step aliases include
environment_step and timesteps; return aliases include avg_return,
eval/mean_reward and avg_reward. Use evaluated policy returns, not training rewards.
All analysis output includes input-file SHA-256 hashes.

## EMG comparison

Export simulated activations with the existing EMG entry points. Annotate each
sample with a cycle column using identified gait events; keep the episode column.
Do not concatenate episodes or use an arbitrary short trajectory as one gait cycle.
Rows in each cycle must be in chronological order and uniformly sampled.

Provide a CSV of a **preprocessed single-cycle human envelope**, one muscle per
column, sampled uniformly over the gait cycle, and a JSON column map such as
`{"soleus_r": "human_soleus", "tibant_r": "human_tibialis"}`.
Acquire human data under its own terms and record dataset version, subject/task,
speed, filtering, envelope extraction and cycle segmentation in --source or a
separate source document. This tool does not turn raw EMG voltage into an envelope.

```bash
python -B -m benchmark_eval.analyze emg --input results/walk_cycles.csv --reference private-data/human_walk_envelope.csv --mapping private-data/muscle_columns.json --source "Dataset version, subject/task/speed and preprocessing record" --output results/emg_similarity.json
```

The implementation resamples each cycle to 101 points, averages cycles, min-max
normalizes per muscle, and searches cyclic shifts maximizing Pearson correlation.
It saves each aligned envelope, spread, shift, score and the mean score. Missing
columns, nonfinite samples and constant envelopes fail explicitly. Human references
for walking, running and stairs must be selected separately.

## Latent dimensions and reconstruction

Train separate encoder/decoder pairs using train_expert_transformer.py and its
--latent-dim option, with fixed train/validation splits and training budgets.
Generate task configurations using matching weights:

```bash
python -B -m rl_paradigms.deprl_middleware_22tasks.generate_configs --latent-dim 32 --strict-weights --encoder-path artifacts/latent32/spinal_encoder_weights.pth --decoder-path artifacts/latent32/spinal_decoder_weights.pth --output-dir artifacts/configs32
```

Repeat for 16, 64 and 128. Use the unmodified 416-action baseline as the full-space
comparison. The dimension is the middleware bottleneck; the existing policy-facing
action space is not changed into a latent-action policy. Missing weights without
--strict-weights invoke the existing pass-through smoke mode, which is **not** a
learned latent baseline. Architecture-mismatched weights must not be substituted.

For held-out expert samples, export matching (sample, actuator) arrays named target
and reconstruction into NPZ, without pickle, then run:

```bash
python -B -m benchmark_eval.analyze reconstruction --input results/heldout_reconstruction.npz --output results/reconstruction.json
```

This reports MSE and variance-weighted explained variance. It does not generate
reconstructions or select a validation split for you.

## Agentic reward tuning

Install `pip install -e ".[agentic]"`. Set MSK_BENCH_AGENTIC_API_KEY (or OPENAI_API_KEY)
and **explicitly** set MSK_BENCH_AGENTIC_MODEL to an available model identifier.
MSK_BENCH_AGENTIC_BASE_URL selects an OpenAI-compatible endpoint. Credentials are
not stored in logs. API calls may incur provider charges; offline tests mock the
client and do not validate access to any named hosted model.

In a copy of the existing depRL AgenticWalk YAML, set the environment expression to:

```python
deprl.environments.Gym('MSKBenchAgenticWalk-v0', scaled_actions=False, enable_agentic=True, agentic_update_steps=20000, agentic_l1_norm=40.0, agentic_weight_dir='artifacts/agentic/model-seed0')
```

Use separate state directories for every model/seed/run. Keep the learner,
initial weights, training budget, evaluation schedule and seeds matched when
comparing providers. The initial weights are normalized once; subsequent proposals
are projected jointly onto fixed L1=40 and +/-20% per-update bounds, preserving
signs and zero weights. All configured reward keys, including penalties, are in the
norm. This scope must be stated when reporting newly run comparisons; historical
norm scope cannot be inferred from the paper alone.

Updates record requested/returned model identifiers, feedback, prompt, proposal
and applied weights in *.updates.jsonl beside the state file. Missing credentials,
missing model configuration, invalid responses and API failures are errors, not
silent successful updates. For smoke tests set enable_agentic=False.
State writes are atomic and updates use an exclusive lock. After a killed process,
remove its stale *.json.lock only after confirming no other writer is active;
locks are not expired automatically during potentially long API requests.
The update interval currently counts each environment's reward evaluations; it is
not a global vectorized learner-step scheduler. A publication claiming a global
20,000-transition update schedule needs trainer-level coordination and its original
configuration. Do not conflate the two protocols.

## Powerlift load and anatomical recruitment

Create an environment with an explicit load:

```python
import gymnasium as gym
import msk_bench
env = gym.make("MSKBenchPowerlift-v0", object_mass_kg=5.0)
try:
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
finally:
    env.close()
```

The load sets dumbbell mass in both simulation models and scales inertia with mass
(fixed geometry), then refreshes MuJoCo constants. The default XML load is preserved
when the argument is omitted. For a load sweep use .05, .5, 5 and 10 kg in separate
runs, with controlled policies/seeds; pass object_mass_kg in a depRL environment
expression to train at each load. Random actions above are only a runtime check.

Export activations of shape (time, actuator) and the exact actuator_names as a
Unicode array in NPZ. Provide an explicit actuator-to-family JSON mapping covering
every actuator once. Do not assume the 416 and 700 models share actuator indices.

```bash
python -B -m benchmark_eval.analyze recruitment --input results/activations.npz --mapping private-data/actuator_families.json --output results/recruitment.json
```

Outputs include mean absolute activation mass and normalized family shares,
plus frame/actuator counts. This dimension-independent analysis supports recordings
from either anatomy. The repository already includes the MS-Human-700 full-body,
locomotion and manipulation XML variants under msk_bench/simhive/ms_human_700/,
and corresponding msgym environment implementations. These configurations are not
missing. The analysis command consumes trajectories and an explicit grouping map;
it does not generate trained policies or automatically reconstruct historical
figure inputs.

## External artifacts for optional experiment replication

The repository supplies 22 canonical tasks, training/evaluation/render entry
points, the existing residual and imitation integrations, and the analysis tools
above. This is a source-code release; weights and experimental recordings do not
need to be bundled. The following describes inputs for reproducing particular
experiments, not missing-file blockers for publishing the source:

- Exact authorized Walk/Run reference data and official pretrained weights remain
  separate downloads (see the root README); pretrained weights are not locally
  retrained substitutes.
- Human EMG comparison code is included. Actual human recordings and preprocessing
  records are separate inputs; CleanEMG_Data.mat referenced by the old plotting
  script was not found in the current source tree.
- MS-Human-700 models and environment code are included. To recreate a specific
  anatomical-study figure, select the corresponding trained policy, trajectories
  and grouping metadata; their absence from a code-only release does not mean the
  700-model configuration is missing.
- Matched agentic historical configurations and global-step coordination are not
  recovered by adding the per-update constraint.
- The GPT direct-control protocol remains file-mediated, with no bundled LLM client
  or historical generated actions; exact past model decisions cannot be recreated
  by a smoke test.
- Original trained checkpoints, seed-wise learning curves and figure input data
  are not included. Plotting new artifacts cannot establish historical equivalence.
