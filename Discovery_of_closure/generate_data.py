import argparse
import os
import itertools
import json
import numpy as np
import pandas as pd

from rad_noise import (
    DEFAULT_NOISE_LEVELS,
    add_noise_suffix,
    case_stem,
    noise_tag,
    parse_noise_level,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Basic setting
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "data")
CASE_DATA_DIR = os.path.join(OUTPUT_DIR, "cases")
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CASE_DATA_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Dimensionless PDE:
# c_t + c_x = (1 / Pe) c_xx + Da * c * (1 - c)
#
# Regression form:
# c_t = -c_x + (1 / Pe) c_xx + Da*c - Da*c^2


# ============================================================
# 2. Given physical variable set
# ============================================================

# You can modify these values.
# The program will generate all combinations of U, L, D, and k.

variable_grid = {
    "U": [0.5, 1.0, 2.0],          # characteristic velocity
    "L": [0.5, 1.0, 2.0],          # characteristic length
    "D": [0.005, 0.01, 0.02],      # diffusion coefficient
    "k": [0.2, 0.5, 1.0],          # reaction rate
}


# Optional filtering range.
# This avoids extremely stiff or numerically difficult cases.
PE_RANGE = (10.0, 400.0)
DA_RANGE = (0.05, 5.0)


# ============================================================
# 3. Numerical grid
# ============================================================

nx = 128
nt_save = 101
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

x = np.linspace(x_min, x_max, nx, endpoint=False)
dx = x[1] - x[0]
t_save = np.linspace(t_min, t_max, nt_save)

# Number of sampled points for PDE discovery from each case
n_samples_per_case = 1000

# Noise level added to c for observation data.
# 0.0 keeps the original clean filenames. Nonzero values append _{tag},
# for example _1pct, _5pct, _10pct, _25pct.
noise_level = 0.00


# ============================================================
# 4. Finite difference operators, periodic boundary
# ============================================================

def periodic_dx(u, dx):
    """First-order spatial derivative with central difference."""
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def periodic_dxx(u, dx):
    """Second-order spatial derivative with central difference."""
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx ** 2)


def rhs_rad(c, pe, da, dx):
    """
    RHS of the dimensionless reaction-advection-diffusion equation.

    c_t = -c_x + (1/Pe)c_xx + Da*c*(1-c)
    """
    c_x = periodic_dx(c, dx)
    c_xx = periodic_dxx(c, dx)

    return -c_x + (1.0 / pe) * c_xx + da * c * (1.0 - c)


def rk4_step(c, dt, pe, da, dx):
    """One RK4 step."""
    k1 = rhs_rad(c, pe, da, dx)
    k2 = rhs_rad(c + 0.5 * dt * k1, pe, da, dx)
    k3 = rhs_rad(c + 0.5 * dt * k2, pe, da, dx)
    k4 = rhs_rad(c + dt * k3, pe, da, dx)

    return c + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================
# 5. Initial condition
# ============================================================

def periodic_gaussian(x, center, width):
    """
    Periodic Gaussian profile.
    This avoids discontinuity at the periodic boundary.
    """
    dist = np.minimum(np.abs(x - center), 1.0 - np.abs(x - center))
    return np.exp(-(dist ** 2) / (2.0 * width ** 2))


def make_initial_condition(x, case_index):
    phase = 0.3 * case_index

    c0 = 0.50
    c0 += 0.25 * np.sin(2.0 * np.pi * x + phase)
    c0 += 0.15 * np.sin(4.0 * np.pi * x + 0.5 * phase)

    # Add a local pulse to enrich nonlinear reaction dynamics
    center = 0.25 + 0.1 * ((case_index % 5) / 4.0)
    width = 0.08
    c0 += 0.15 * periodic_gaussian(x, center=center, width=width)

    c0 = np.clip(c0, 0.05, 0.95)

    return c0

# ============================================================
# 6. Parameter generation
# ============================================================

def generate_parameter_cases(variable_grid):
    """
    Generate all combinations of U, L, D, and k.
    Compute Pe and Da for each case.
    """
    keys = list(variable_grid.keys())
    values = [variable_grid[k] for k in keys]

    records = []
    case_id = 0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        U = params["U"]
        L = params["L"]
        D = params["D"]
        k = params["k"]

        pe = U * L / D
        da = k * L / U

        if not (PE_RANGE[0] <= pe <= PE_RANGE[1]):
            continue

        if not (DA_RANGE[0] <= da <= DA_RANGE[1]):
            continue

        record = {
            "case_id": case_id,
            "U": U,
            "L": L,
            "D": D,
            "k": k,
            "Pe": pe,
            "Da": da,
            "coef_cx_true": -1.0,
            "coef_cxx_true": 1.0 / pe,
            "coef_c_true": da,
            "coef_c2_true": -da,
        }

        records.append(record)
        case_id += 1

    return pd.DataFrame(records)


def row_to_metadata(row):
    """
    Convert a pandas row to a JSON-serializable metadata dictionary.
    """
    metadata = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            metadata[key] = int(value)
        elif isinstance(value, (np.floating,)):
            metadata[key] = float(value)
        else:
            metadata[key] = value
    return metadata


# ============================================================
# 7. PDE solver
# ============================================================

def solve_one_case(pe, da, case_index):
    """
    Solve one case and return c(t, x).
    """
    c = make_initial_condition(x, case_index)

    C_save = np.zeros((nt_save, nx), dtype=np.float64)
    C_save[0] = c.copy()

    # Conservative stable time step for explicit method.
    # For diffusion: dt <= dx^2 / (2 * diffusion coefficient)
    # Here diffusion coefficient is 1 / Pe.
    diffusion = 1.0 / pe
    dt_diff = 0.25 * dx * dx / max(diffusion, 1e-12)
    dt_adv = 0.25 * dx
    dt_react = 0.10 / max(da, 1e-12)

    dt_internal = min(dt_diff, dt_adv, dt_react, 1e-3)

    current_t = t_min

    for j in range(1, nt_save):
        target_t = t_save[j]

        while current_t < target_t - 1e-12:
            dt = min(dt_internal, target_t - current_t)
            c = rk4_step(c, dt, pe, da, dx)
            c = np.clip(c, -0.5, 1.5)
            current_t += dt

        C_save[j] = c.copy()

    return C_save


def add_observation_noise(C_clean, level, seed=RANDOM_SEED):
    """Add Gaussian noise with amplitude level * std(C_clean)."""
    C_clean = np.asarray(C_clean, dtype=np.float64)
    level = float(level)
    if level <= 1e-15:
        return C_clean.copy()
    rng = np.random.RandomState(int(seed) + int(round(level * 1000.0)))
    std_c = float(np.std(C_clean))
    if std_c <= 0:
        return C_clean.copy()
    return C_clean + level * std_c * rng.randn(*C_clean.shape)


def save_one_case_dataset(row, C_clean, C_observed, output_dir=CASE_DATA_DIR, tag=""):
    """
    Save one case dataset and its corresponding physical/dimensionless variables.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = row_to_metadata(row)
    case_id = int(metadata["case_id"])
    stem = case_stem(case_id, tag)

    npz_path = os.path.join(output_dir, "{}.npz".format(stem))
    metadata_path = os.path.join(output_dir, "{}_variables.json".format(stem))

    np.savez_compressed(
        npz_path,
        C_clean=C_clean,
        C_observed=C_observed,
        x=x,
        t=t_save,
        U=float(metadata["U"]),
        L=float(metadata["L"]),
        D=float(metadata["D"]),
        k=float(metadata["k"]),
        Pe=float(metadata["Pe"]),
        Da=float(metadata["Da"]),
        coef_cx_true=float(metadata["coef_cx_true"]),
        coef_cxx_true=float(metadata["coef_cxx_true"]),
        coef_c_true=float(metadata["coef_c_true"]),
        coef_c2_true=float(metadata["coef_c2_true"]),
        noise_level=float(metadata.get("noise_level", 0.0)),
        noise_tag=str(metadata.get("noise_tag", tag)),
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npz_path, metadata_path


def save_case_datasets(params_df, all_C, all_C_observed, output_dir=CASE_DATA_DIR, tag=""):
    """
    Save every solved case as an independent dataset under data/cases.
    """
    saved_records = []
    for case_idx, row in params_df.iterrows():
        npz_path, metadata_path = save_one_case_dataset(
            row,
            all_C[case_idx],
            all_C_observed[case_idx],
            output_dir=output_dir,
            tag=tag,
        )
        saved_records.append(
            {
                "case_id": int(row["case_id"]),
                "dataset_path": npz_path,
                "variables_path": metadata_path,
                "noise_tag": tag,
            }
        )

    index_df = pd.DataFrame(saved_records)
    index_path = os.path.join(OUTPUT_DIR, add_noise_suffix("case_file_index.csv", tag))
    index_df.to_csv(index_path, index=False)

    print("\nCase datasets saved:")
    print(output_dir)
    print(index_path)

    return index_df


def configure_nature_heatmap_style():
    """
    Compact journal-style matplotlib settings for heatmaps.
    """
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def finish_nature_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=2)


def plot_case_heatmap(row, C_case, save_dir=HEATMAP_DIR, field_label="c", tag=""):
    """
    Plot one case as a Nature-style heatmap in the t-x plane.
    """
    os.makedirs(save_dir, exist_ok=True)
    configure_nature_heatmap_style()

    metadata = row_to_metadata(row)
    case_id = int(metadata["case_id"])

    fig, ax = plt.subplots(figsize=(3.35, 2.45), dpi=300)
    image = ax.imshow(
        C_case,
        origin="lower",
        extent=[x_min, x_max, t_min, t_max],
        aspect="auto",
        cmap="viridis",
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(
        "Case {:03d}  Pe={:.2f}, Da={:.2f}".format(
            case_id,
            float(metadata["Pe"]),
            float(metadata["Da"]),
        )
    )
    finish_nature_axes(ax)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(field_label)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=2.5, width=0.6, pad=2)

    png_path = os.path.join(save_dir, add_noise_suffix("case_{:03d}_heatmap.png".format(case_id), tag))
    pdf_path = os.path.join(save_dir, add_noise_suffix("case_{:03d}_heatmap.pdf".format(case_id), tag))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def plot_all_case_heatmaps(params_df, all_C_observed, save_dir=HEATMAP_DIR, tag=""):
    """
    Plot heatmaps for all solved cases.
    """
    saved_paths = []
    for case_idx, row in params_df.iterrows():
        saved_paths.append(
            plot_case_heatmap(row, all_C_observed[case_idx], save_dir=save_dir, tag=tag)
        )

    print("\nCase heatmaps saved:")
    print(save_dir)

    return saved_paths


# ============================================================
# 8. Generate full field dataset
# ============================================================

def load_clean_fields():
    npz_path = os.path.join(OUTPUT_DIR, "full_field_dataset.npz")
    params_path = os.path.join(OUTPUT_DIR, "params.csv")
    if not (os.path.exists(npz_path) and os.path.exists(params_path)):
        return None, None
    payload = np.load(npz_path, allow_pickle=True)
    if "C_clean" not in payload:
        return None, None
    params_df = pd.read_csv(params_path)
    return params_df, np.asarray(payload["C_clean"], dtype=np.float64)


def solve_clean_fields():
    params_df = generate_parameter_cases(variable_grid)
    if len(params_df) == 0:
        raise ValueError("No valid parameter cases. Please adjust variable_grid, PE_RANGE, or DA_RANGE.")

    all_C = []
    for _, row in params_df.iterrows():
        case_id = int(row["case_id"])
        pe = float(row["Pe"])
        da = float(row["Da"])
        print("Solving case {:03d}: Pe={:.3f}, Da={:.3f}".format(case_id, pe, da))
        all_C.append(solve_one_case(pe, da, case_id))
    return params_df, np.array(all_C, dtype=np.float64)


def generate_full_dataset(level=None, reuse_clean=True):
    level = parse_noise_level(noise_level if level is None else level)
    tag = noise_tag(level)

    params_df = None
    all_C = None
    if reuse_clean:
        params_df, all_C = load_clean_fields()
        if all_C is not None:
            print("Reusing clean fields from full_field_dataset.npz")

    if all_C is None:
        params_df, all_C = solve_clean_fields()
        if level > 0:
            clean_npz = os.path.join(OUTPUT_DIR, "full_field_dataset.npz")
            params_path = os.path.join(OUTPUT_DIR, "params.csv")
            params_df.to_csv(params_path, index=False)
            np.savez(
                clean_npz,
                C_clean=all_C,
                C_observed=all_C,
                x=x,
                t=t_save,
                params=params_df.to_records(index=False),
                noise_level=0.0,
            )
            print("\nClean full field dataset saved:")
            print(clean_npz)

    all_C_observed = add_observation_noise(all_C, level)
    params_df = params_df.copy()
    if tag:
        params_df["noise_level"] = level
        params_df["noise_tag"] = tag

    params_path = os.path.join(OUTPUT_DIR, add_noise_suffix("params.csv", tag))
    npz_path = os.path.join(OUTPUT_DIR, add_noise_suffix("full_field_dataset.npz", tag))
    params_df.to_csv(params_path, index=False)
    np.savez(
        npz_path,
        C_clean=all_C,
        C_observed=all_C_observed,
        x=x,
        t=t_save,
        params=params_df.to_records(index=False),
        noise_level=level,
        noise_tag=tag,
    )

    print("\nFull field dataset saved:")
    print(npz_path)
    print(params_path)

    save_case_datasets(params_df, all_C, all_C_observed, tag=tag)
    return params_df, all_C, all_C_observed, tag


# ============================================================
# 9. Generate samples for PDE discovery
# ============================================================

def generate_discovery_samples(params_df, all_C_observed, tag=""):
    """
    Generate pointwise samples for PDE discovery.

    Target:
        c_t

    Candidate terms:
        c_x, c_xx, c, c^2

    True structure:
        c_t = -c_x + (1/Pe)c_xx + Da*c - Da*c^2
    """
    records = []

    for case_idx, row in params_df.iterrows():
        case_id = int(row["case_id"])
        pe = float(row["Pe"])
        da = float(row["Da"])

        C = all_C_observed[case_idx]

        # Avoid the first and last saved time index when sampling.
        # Spatial derivatives use periodic boundary, so all x indices are allowed.
        valid_t_indices = np.arange(1, nt_save - 1)
        valid_x_indices = np.arange(0, nx)

        total_points = len(valid_t_indices) * len(valid_x_indices)
        sample_indices = np.random.choice(
            total_points,
            size=min(n_samples_per_case, total_points),
            replace=False,
        )

        for idx in sample_indices:
            local_t_idx = idx // nx
            ix = idx % nx
            it = valid_t_indices[local_t_idx]

            c_snapshot = C[it]
            c_value = c_snapshot[ix]

            c_x_snapshot = periodic_dx(c_snapshot, dx)
            c_xx_snapshot = periodic_dxx(c_snapshot, dx)

            c_x_value = c_x_snapshot[ix]
            c_xx_value = c_xx_snapshot[ix]

            # Use the PDE itself to compute clean c_t from current observed field.
            # This avoids unstable finite-difference estimation in time.
            c_t_value = (
                -c_x_value
                + (1.0 / pe) * c_xx_value
                + da * c_value * (1.0 - c_value)
            )

            records.append({
                "case_id": case_id,
                "x": x[ix],
                "t": t_save[it],
                "c": c_value,
                "c_t": c_t_value,
                "c_x": c_x_value,
                "c_xx": c_xx_value,
                "c2": c_value ** 2,

                "U": row["U"],
                "L": row["L"],
                "D": row["D"],
                "k": row["k"],

                "Pe": pe,
                "Da": da,

                "coef_cx_true": -1.0,
                "coef_cxx_true": 1.0 / pe,
                "coef_c_true": da,
                "coef_c2_true": -da,
            })

    sample_df = pd.DataFrame(records)

    sample_path = os.path.join(OUTPUT_DIR, add_noise_suffix("discovery_samples.csv", tag))
    sample_df.to_csv(sample_path, index=False)

    print("\nDiscovery samples saved:")
    print(sample_path)

    return sample_df


# ============================================================
# 10. Main
# ============================================================

def coerce_noise_levels(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [parse_noise_level(item) for item in value]
    return [parse_noise_level(value)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RAD fields. Nonzero --noise-level writes _{tag} files and leaves clean data untouched."
    )
    parser.add_argument(
        "--noise-level",
        action="append",
        dest="noise_levels",
        default=None,
        help="Relative Gaussian noise. Repeat to write several levels, e.g. --noise-level 0.01 --noise-level 0.05.",
    )
    parser.add_argument(
        "--all-noise-levels",
        action="store_true",
        help="Generate 1%, 5%, 10%, and 25% in one run.",
    )
    parser.add_argument(
        "--no-reuse-clean",
        action="store_true",
        help="Re-solve the PDE even if data/full_field_dataset.npz already exists.",
    )
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--no-discovery-samples", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all_noise_levels:
        levels = list(DEFAULT_NOISE_LEVELS)
    else:
        levels = coerce_noise_levels(args.noise_levels)
        if not levels:
            levels = [parse_noise_level(noise_level)]

    for level in levels:
        tag = noise_tag(level)
        print("\n=== noise_level={} tag={} ===".format(level, tag or "clean"))
        params_df, all_C, all_C_observed, tag = generate_full_dataset(
            level=level,
            reuse_clean=not args.no_reuse_clean,
        )
        if not args.no_discovery_samples:
            sample_df = generate_discovery_samples(params_df, all_C_observed, tag=tag)
        else:
            sample_df = None
        if not args.no_heatmaps:
            plot_all_case_heatmaps(params_df, all_C_observed, tag=tag)

        print("\nParameter cases:")
        print(params_df.head())
        if sample_df is not None:
            print("\nDiscovery samples:")
            print(sample_df.head())

    print("\nDataset generation finished.")
