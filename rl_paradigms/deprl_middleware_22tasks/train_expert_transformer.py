from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from deprl_middleware_22tasks.networks import FullAnatomicalTransformer, build_encoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train middleware encoder/decoder from expert synergy data.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = torch.load(args.data, map_location="cpu")
    dataset = TensorDataset(data["priors"], data["states"], data["moments"], data["actions"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    num_muscles = int(data["states"].shape[1])
    moments = data["moments"]
    num_joints = int(moments.shape[-1]) if moments.dim() >= 3 else int(moments.shape[-1] // num_muscles)

    encoder = build_encoder(num_muscles, args.latent_dim).to(device)
    decoder = FullAnatomicalTransformer(args.latent_dim, num_muscles, num_joints).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for priors, states, moments, actions in loader:
            priors = priors.to(device)
            states = states.to(device)
            moments = moments.to(device)
            actions = actions.to(device)
            optimizer.zero_grad()
            latent = encoder(actions)
            reconstructed = decoder(latent, priors, states, moments)
            loss = criterion(reconstructed, (actions + 1.0) / 2.0)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        print(f"epoch {epoch + 1:03d}/{args.epochs:03d} mse={total_loss / max(len(loader), 1):.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = args.output_dir / "spinal_encoder_weights.pth"
    decoder_path = args.output_dir / "spinal_decoder_weights.pth"
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"saved encoder: {encoder_path}")
    print(f"saved decoder: {decoder_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
