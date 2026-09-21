"""Real expert-data collection and auxiliary Transformer training/reload."""

import json

from test_deprl_training_runtime import run_isolated, test_short_train_updates_saves_and_reloads as train_deprl


def test_expert_collection_transformer_training_and_reload(tmp_path):
    train_deprl(tmp_path, "deprl", 1)
    manifest = next((tmp_path / "runs").rglob("training_manifest.json"))
    assert json.loads(manifest.read_text())["algorithm"] == "deprl"
    data = tmp_path / "expert.pt"
    output = tmp_path / "transformer"
    run_isolated([
        "-m", "rl_paradigms.deprl_middleware_22tasks.collect_expert_data",
        "--env", "MSKBenchSquat-v0", "--checkpoint-dir", str(manifest.parent),
        "--samples", "4", "--output", str(data),
    ], tmp_path)
    run_isolated([
        "-m", "rl_paradigms.deprl_middleware_22tasks.train_expert_transformer",
        "--data", str(data), "--output-dir", str(output), "--epochs", "1",
        "--batch-size", "2", "--latent-dim", "8", "--device", "cpu",
    ], tmp_path)
    for name in ("spinal_encoder_weights.pth", "spinal_decoder_weights.pth"):
        assert (output / name).stat().st_size > 0
    run_isolated(["-c", f"""
import torch
from pathlib import Path
from deprl_middleware_22tasks.networks import build_encoder, FullAnatomicalTransformer
data = torch.load({str(data)!r}, map_location='cpu', weights_only=True)
assert all(torch.isfinite(value).all() and len(value) == 4 for value in data.values())
muscles, joints = data['states'].shape[1], data['moments'].shape[-1]
encoder = build_encoder(muscles, 8)
decoder = FullAnatomicalTransformer(8, muscles, joints)
directory = Path({str(output)!r})
encoder.load_state_dict(torch.load(directory / 'spinal_encoder_weights.pth', weights_only=True))
decoder.load_state_dict(torch.load(directory / 'spinal_decoder_weights.pth', weights_only=True))
encoder.eval()
decoder.eval()
with torch.no_grad():
    result = decoder(encoder(data['actions']), data['priors'], data['states'], data['moments'])
assert result.shape == data['actions'].shape
assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
"""], tmp_path)
