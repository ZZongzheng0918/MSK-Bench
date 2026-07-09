# MSK-Bench depRL Training Configs

Run a task with:

```bash
python -m deprl.main experiments/msk_bench_training_files/msk_bench_powerlift.yaml
```

Each config imports `msk_bench` in the header so the `MSKBench{Name}-v0` Gymnasium environments are registered before depRL builds the environment.

Config files use `msk_bench_<task_name>.yaml`, where `<task_name>` is the snake_case form of the environment task name. For example, `MSKBenchWalkTurn-v0` uses `msk_bench_walk_turn.yaml`.