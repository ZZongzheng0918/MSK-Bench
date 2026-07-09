# SAC for MSK-Bench

Stable-Baselines3 SAC benchmark trainer for the 22 MSK-Bench Gymnasium environments.

```powershell
python train_sac_msk_bench.py --list-envs
python train_sac_msk_bench.py --env MSKBenchStand-v0 --num-envs 8
python train_sac_msk_bench.py --env all --dry-run
```

Outputs are written under `runs/<task_name>/` by default.