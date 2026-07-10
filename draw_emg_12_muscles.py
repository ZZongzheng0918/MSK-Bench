"""Plot 12 target-muscle EMG trajectories from exported benchmark CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from emg_export_common import TARGET_EMG_MUSCLES

MUSCLE_MAPPING = {
    "soleus_r": "SoleusMedialis",
    "soleus_l": "SoleusLateralis",
    "gasmed_r": "GastrocnemuisMedialis",
    "gaslat_r": "GastrocnemiusLateralis",
    "tibant_r": "TibialisAnterior",
    "perlong_r": "PeroneusLongus",
    "perbrev_r": "PeroneusBrevis",
    "recfem_r": "RectusFemoris",
    "vaslat_r": "VastusLateralis",
    "vasmed_r": "VastusMedialis",
    "bflh_r": "BicepsFemoris",
    "semiten_r": "Semitendinosus",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw a 2x6 EMG comparison figure for the 12 target muscles.")
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Add one exported EMG CSV series. Example: --series PPO=ppo_walk_episode1_emg.csv",
    )
    parser.add_argument("--human-mat", type=Path, default=None, help="Optional CleanEMG_Data.mat human EMG reference.")
    parser.add_argument("--output-pdf", type=Path, default=Path("EMG_12_Muscles_2x6_CoRL.pdf"))
    parser.add_argument("--output-png", type=Path, default=Path("EMG_12_Muscles_2x6_CoRL.png"))
    return parser


def normalize_to_gait_cycle(data_array):
    import numpy as np
    from scipy.interpolate import interp1d

    data_array = np.asarray(data_array, dtype=float).reshape(-1)
    data_array = data_array[np.isfinite(data_array)]
    if data_array.size == 0:
        return np.zeros(101, dtype=float)
    if data_array.size == 1:
        return np.full(101, float(data_array[0]), dtype=float)
    x_old = np.linspace(0, 1, data_array.size)
    x_new = np.linspace(0, 1, 101)
    kind = "cubic" if data_array.size >= 4 else "linear"
    return interp1d(x_old, data_array, kind=kind)(x_new)


def min_max_normalize(data_array):
    import numpy as np

    data_array = np.asarray(data_array, dtype=float)
    d_min, d_max = np.nanmin(data_array), np.nanmax(data_array)
    return (data_array - d_min) / (d_max - d_min + 1e-8)


def format_r(value: float) -> str:
    if value <= 0:
        return "0.00"
    if value >= 1.0:
        return "1.00"
    return f"{value:.2f}"


def process_and_align_sim(raw_signal, real_norm=None):
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.stats import pearsonr

    raw_signal = np.asarray(raw_signal, dtype=float).reshape(-1)
    raw_signal = raw_signal[np.isfinite(raw_signal)]
    if raw_signal.size == 0:
        return None, None, 0.0
    prominence = float(np.max(raw_signal) * 0.15) if raw_signal.size else 0.0
    peaks, _ = find_peaks(raw_signal, distance=60, prominence=prominence)

    if len(peaks) < 2:
        sim_mean_raw = normalize_to_gait_cycle(raw_signal)
        sim_norm = min_max_normalize(sim_mean_raw)
        sim_std = np.zeros_like(sim_norm)
    else:
        cycles = [normalize_to_gait_cycle(raw_signal[peaks[i] : peaks[i + 1]]) for i in range(len(peaks) - 1)]
        sim_cycles = np.asarray(cycles, dtype=float)
        sim_mean_raw = np.mean(sim_cycles, axis=0)
        sim_norm = min_max_normalize(sim_mean_raw)
        sim_scale = float(np.max(sim_mean_raw) - np.min(sim_mean_raw))
        sim_std = np.std(sim_cycles, axis=0) / (sim_scale if sim_scale != 0 else 1)

    if real_norm is None:
        return sim_norm, sim_std, 0.0

    best_r = -1.0
    aligned_mean, aligned_std = sim_norm, sim_std
    for shift in range(101):
        shifted_mean = np.roll(sim_norm, shift)
        if np.std(shifted_mean) < 1e-6 or np.std(real_norm) < 1e-6:
            r_value = 0.0
        else:
            r_value, _ = pearsonr(shifted_mean, real_norm)
        if r_value > best_r:
            best_r = float(r_value)
            aligned_mean = shifted_mean
            aligned_std = np.roll(sim_std, shift)
    return aligned_mean, aligned_std, best_r


def parse_series(values: list[str]):
    import pandas as pd

    series = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"--series must be LABEL=CSV, got: {item}")
        label, path_text = item.split("=", 1)
        label = label.strip()
        path = Path(path_text.strip())
        if not label:
            raise ValueError(f"Missing label in --series {item}")
        if not path.exists():
            raise FileNotFoundError(path)
        series.append((label, pd.read_csv(path)))
    return series


def load_human_steps(mat_path: Path):
    import scipy.io as sio

    mat_data = sio.loadmat(mat_path)
    return mat_data["CleanEMG"].flat[0]


def human_muscle_cycles(clean_emg, mat_name: str):
    cycles = []
    for t_name in clean_emg.dtype.names:
        trial = clean_emg[t_name].flat[0]
        for s_name in trial.dtype.names:
            step = trial[s_name].flat[0]
            if mat_name in step.dtype.names:
                cycles.append(normalize_to_gait_cycle(step[mat_name].flatten()))
    return cycles


def short_muscle_name(mat_name: str) -> str:
    return (
        mat_name.replace("Gastrocnemuis", "Gas. ")
        .replace("Gastrocnemius", "Gas. ")
        .replace("Medialis", "Med.")
        .replace("Lateralis", "Lat.")
        .replace("Anterior", "Ant.")
        .replace("Femoris", "Fem.")
        .replace("Rectus", "Rec. ")
        .replace("Vastus", "Vas. ")
        .replace("Biceps", "Bic. ")
        .replace("Peroneus", "Per. ")
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.series:
        parser.error("at least one --series LABEL=CSV is required")

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = parse_series(args.series)
    clean_emg = load_human_steps(args.human_mat) if args.human_mat else None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
        }
    )

    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(8.5, 3.2))
    axes_flat = axes.flatten()
    x = np.linspace(0, 100, 101)
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]

    for idx, csv_name in enumerate(TARGET_EMG_MUSCLES):
        ax = axes_flat[idx]
        mat_name = MUSCLE_MAPPING[csv_name]
        real_norm, real_std = None, None

        if clean_emg is not None:
            human_cycles = human_muscle_cycles(clean_emg, mat_name)
            if human_cycles:
                real_mean_raw = np.mean(np.asarray(human_cycles, dtype=float), axis=0)
                real_norm = min_max_normalize(real_mean_raw)
                real_scale = float(np.max(real_mean_raw) - np.min(real_mean_raw))
                real_std = np.std(np.asarray(human_cycles, dtype=float), axis=0) / (real_scale if real_scale != 0 else 1)
                ax.plot(x, real_norm, color="#d62728", linestyle="--", label="Human EMG" if idx == 0 else "")
                ax.fill_between(x, real_norm - real_std, real_norm + real_std, color="#d62728", alpha=0.15)

        r_values = {}
        for dataset_index, (label, frame) in enumerate(datasets):
            if csv_name not in frame.columns:
                r_values[label] = 0.0
                continue
            signal = frame[csv_name].to_numpy(dtype=float, na_value=np.nan)
            sim_norm, sim_std, r_value = process_and_align_sim(signal, real_norm)
            if sim_norm is None:
                r_values[label] = 0.0
                continue
            color = palette[dataset_index % len(palette)]
            ax.plot(x, sim_norm, color=color, label=label if idx == 0 else "")
            ax.fill_between(x, sim_norm - sim_std, sim_norm + sim_std, color=color, alpha=0.15)
            r_values[label] = r_value

        title = short_muscle_name(mat_name)
        if real_norm is not None and r_values:
            r_chunks = [f"{label}: {format_r(value)}" for label, value in r_values.items()]
            title = f"{title}\n" + "  ".join(r_chunks[:2])
            if len(r_chunks) > 2:
                title += "\n" + "  ".join(r_chunks[2:4])
        elif real_norm is None:
            title = f"{title}\n(Sim Only)"
        ax.set_title(title, pad=3, fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"])
        if idx % 6 == 0:
            ax.set_ylabel("Norm. Act.")
        if idx >= 6:
            ax.set_xlabel("Gait (%)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    handles_by_label = {}
    for ax in axes_flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in handles_by_label:
                handles_by_label[label] = handle
    labels = list(handles_by_label)
    handles = [handles_by_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(max(len(labels), 1), 5),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.5, h_pad=1.0)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_pdf, transparent=True)
    plt.savefig(args.output_png, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    print(f"saved PDF: {args.output_pdf}")
    print(f"saved PNG: {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
