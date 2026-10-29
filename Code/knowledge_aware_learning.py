import json
import math
import random
import itertools
from pathlib import Path
from functools import reduce
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Basic settings
# ============================================================

BASE_DIMS = ["M", "L", "T", "I", "Theta", "N", "J"]

# This single variable controls prior exclusion and the case-specific model.
# Use None (or "global") to train/load the default model.
training_case = "benard"  #{"benard", "colebrook", "rayleigh", "mhd", "blasius"}

ALPHA_VALUES = [-3, -2, -1, 0, 1, 2, 3]
ALPHA_TO_IDX = {a: i for i, a in enumerate(ALPHA_VALUES)}
IDX_TO_ALPHA = {i: a for a, i in ALPHA_TO_IDX.items()}


def normalize_alpha_value(value, default=None):
    """
    Convert alpha values from JSON/training data into a stable numeric key.

    JSON files may store fractional exponents as strings such as "1/2".
    Integers are kept as int so existing labels like 1 and -1 keep the same
    representation. Non-integer values are stored as Fraction to avoid float
    equality surprises in ALPHA_TO_IDX.
    """
    if value is None:
        return default

    try:
        if isinstance(value, Fraction):
            alpha = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            text = text.replace("−", "-").replace(" ", "")
            alpha = Fraction(text)
        elif isinstance(value, (int, np.integer)):
            alpha = Fraction(int(value), 1)
        elif isinstance(value, (float, np.floating)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Invalid alpha value: {value!r}")
            alpha = Fraction(numeric).limit_denominator(1000000)
        else:
            alpha = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid alpha value: {value!r}") from exc

    if alpha.denominator == 1:
        return alpha.numerator
    return alpha


def normalize_numeric_value(value, default=None):
    return normalize_alpha_value(value, default=default)


def numeric_to_fraction(value, default=0):
    normalized = normalize_numeric_value(value, default=default)
    if isinstance(normalized, Fraction):
        return normalized
    return Fraction(normalized, 1)


def format_alpha_value(value):
    alpha = normalize_alpha_value(value, default=0)
    if isinstance(alpha, Fraction):
        return f"{alpha.numerator}/{alpha.denominator}"
    return str(alpha)


def format_numeric_value(value):
    normalized = normalize_numeric_value(value, default=0)
    if isinstance(normalized, Fraction):
        return f"{normalized.numerator}/{normalized.denominator}"
    return str(normalized)


def set_alpha_values(values):
    global ALPHA_VALUES, ALPHA_TO_IDX, IDX_TO_ALPHA

    normalized = []
    for value in values:
        alpha = normalize_alpha_value(value)
        if alpha is not None and alpha not in normalized:
            normalized.append(alpha)

    ALPHA_VALUES = sorted(normalized)
    ALPHA_TO_IDX = {a: i for i, a in enumerate(ALPHA_VALUES)}
    IDX_TO_ALPHA = {i: a for a, i in ALPHA_TO_IDX.items()}


def extend_alpha_values_from_samples(samples):
    values = list(ALPHA_VALUES)
    for sample in samples:
        for var in sample.get("variables", []):
            alpha = normalize_alpha_value(var.get("alpha"))
            if alpha is not None:
                values.append(alpha)
    set_alpha_values(values)
    return ALPHA_VALUES


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dim(M=0, L=0, T=0, I=0, Theta=0, N=0, J=0):
    return {
        "M": M,
        "L": L,
        "T": T,
        "I": I,
        "Theta": Theta,
        "N": N,
        "J": J,
    }


def make_var(symbol, name, unit, dimension_vector, quantity_type, role, alpha=None):
    item = {
        "symbol": symbol,
        "name": name,
        "unit": unit,
        "dimension_vector": dimension_vector,
        "quantity_type": quantity_type,
        "role": role,
    }
    if alpha is not None:
        item["alpha"] = alpha
    return item


def dimension_matrix(sample):
    variables = sample["variables"]
    D = np.array(
        [
            [v["dimension_vector"].get(base, 0) for v in variables]
            for base in BASE_DIMS
        ],
        dtype=float,
    )
    return D


def alpha_vector(sample):
    return np.array(
        [float(normalize_alpha_value(v.get("alpha"), default=0)) for v in sample["variables"]],
        dtype=float,
    )


def validate_sample(sample):
    D = dimension_matrix(sample)
    a = alpha_vector(sample)
    residual = D @ a
    if not np.allclose(residual, 0.0, atol=1e-10):
        raise ValueError(
            f"Sample {sample['id']} is not dimensionless. D@alpha={residual}"
        )


def is_dimensionless_sample(sample, atol=1e-10):
    D = dimension_matrix(sample)
    a = alpha_vector(sample)
    return np.allclose(D @ a, 0.0, atol=atol)


# ============================================================
# 2. Build a small toy database
# ============================================================

def build_toy_database(path="dimensionless_toy_database.jsonl"):
    path = Path(path)

    density = dim(M=1, L=-3)
    velocity = dim(L=1, T=-1)
    length = dim(L=1)
    dyn_viscosity = dim(M=1, L=-1, T=-1)
    diffusivity = dim(L=2, T=-1)
    pressure = dim(M=1, L=-1, T=-2)
    surface_tension = dim(M=1, T=-2)
    gravity = dim(L=1, T=-2)
    thermal_conductivity = dim(M=1, L=1, T=-3, Theta=-1)
    heat_transfer_coeff = dim(M=1, T=-3, Theta=-1)
    specific_heat = dim(L=2, T=-2, Theta=-1)
    time_dim = dim(T=1)
    temp_dim = dim(Theta=1)

    samples = []

    def add_sample(id_, name, domain, meaning, formula, variables):
        sample = {
            "id": id_,
            "name": name,
            "domain": domain,
            "physical_meaning": meaning,
            "formula": formula,
            "variables": variables,
        }
        validate_sample(sample)
        samples.append(sample)

    # 1. Reynolds number: Re = rho v L / mu
    add_sample(
        "dimless_0001",
        "Reynolds number",
        "fluid_mechanics",
        "ratio of inertial to viscous effects",
        "rho * v * L / mu",
        [
            make_var("rho", "density", "kg m^-3", density, "density", "material_property", 1),
            make_var("v", "velocity", "m s^-1", velocity, "velocity", "flow_scale", 1),
            make_var("L", "length scale", "m", length, "length", "geometric_scale", 1),
            make_var("mu", "dynamic viscosity", "kg m^-1 s^-1", dyn_viscosity, "viscosity", "material_property", -1),
            make_var("T0", "temperature", "K", temp_dim, "temperature", "unused_context", 0),
        ],
    )

    # 2. Peclet number: Pe = v L / D
    add_sample(
        "dimless_0002",
        "Peclet number",
        "transport",
        "ratio of advective to diffusive transport",
        "v * L / D",
        [
            make_var("v", "velocity", "m s^-1", velocity, "velocity", "transport_scale", 1),
            make_var("L", "length scale", "m", length, "length", "geometric_scale", 1),
            make_var("D", "diffusivity", "m^2 s^-1", diffusivity, "diffusivity", "material_property", -1),
            make_var("rho", "density", "kg m^-3", density, "density", "unused_context", 0),
        ],
    )

    # 3. Prandtl number: Pr = mu cp / k
    add_sample(
        "dimless_0003",
        "Prandtl number",
        "heat_transfer",
        "ratio of momentum diffusivity to thermal diffusivity",
        "mu * cp / k",
        [
            make_var("mu", "dynamic viscosity", "kg m^-1 s^-1", dyn_viscosity, "viscosity", "material_property", 1),
            make_var("cp", "specific heat capacity", "m^2 s^-2 K^-1", specific_heat, "specific_heat", "material_property", 1),
            make_var("k", "thermal conductivity", "kg m s^-3 K^-1", thermal_conductivity, "thermal_conductivity", "material_property", -1),
            make_var("L", "length scale", "m", length, "length", "unused_context", 0),
        ],
    )

    # 4. Schmidt number: Sc = mu / (rho D)
    add_sample(
        "dimless_0004",
        "Schmidt number",
        "mass_transfer",
        "ratio of momentum diffusivity to mass diffusivity",
        "mu / (rho * D)",
        [
            make_var("mu", "dynamic viscosity", "kg m^-1 s^-1", dyn_viscosity, "viscosity", "material_property", 1),
            make_var("rho", "density", "kg m^-3", density, "density", "material_property", -1),
            make_var("D", "mass diffusivity", "m^2 s^-1", diffusivity, "diffusivity", "material_property", -1),
            make_var("T0", "temperature", "K", temp_dim, "temperature", "unused_context", 0),
        ],
    )

    # 5. Weber number: We = rho v^2 L / sigma
    add_sample(
        "dimless_0005",
        "Weber number",
        "fluid_mechanics",
        "ratio of inertial to surface tension effects",
        "rho * v^2 * L / sigma",
        [
            make_var("rho", "density", "kg m^-3", density, "density", "material_property", 1),
            make_var("v", "velocity", "m s^-1", velocity, "velocity", "flow_scale", 2),
            make_var("L", "length scale", "m", length, "length", "geometric_scale", 1),
            make_var("sigma", "surface tension", "kg s^-2", surface_tension, "surface_tension", "interface_property", -1),
            make_var("T0", "temperature", "K", temp_dim, "temperature", "unused_context", 0),
        ],
    )

    # 6. Squared Froude number: Fr^2 = v^2 / (g L)
    add_sample(
        "dimless_0006",
        "Squared Froude number",
        "fluid_mechanics",
        "ratio of inertial to gravitational effects",
        "v^2 / (g * L)",
        [
            make_var("v", "velocity", "m s^-1", velocity, "velocity", "flow_scale", 2),
            make_var("g", "gravitational acceleration", "m s^-2", gravity, "acceleration", "body_force_scale", -1),
            make_var("L", "length scale", "m", length, "length", "geometric_scale", -1),
            make_var("rho", "density", "kg m^-3", density, "density", "unused_context", 0),
        ],
    )

    # 7. Euler number: Eu = p / (rho v^2)
    add_sample(
        "dimless_0007",
        "Euler number",
        "fluid_mechanics",
        "ratio of pressure to inertial effects",
        "p / (rho * v^2)",
        [
            make_var("p", "pressure", "kg m^-1 s^-2", pressure, "pressure", "driving_force", 1),
            make_var("rho", "density", "kg m^-3", density, "density", "material_property", -1),
            make_var("v", "velocity", "m s^-1", velocity, "velocity", "flow_scale", -2),
            make_var("L", "length scale", "m", length, "length", "unused_context", 0),
        ],
    )

    # 8. Biot number: Bi = h L / k
    add_sample(
        "dimless_0008",
        "Biot number",
        "heat_transfer",
        "ratio of internal to surface thermal resistance",
        "h * L / k",
        [
            make_var("h", "heat transfer coefficient", "kg s^-3 K^-1", heat_transfer_coeff, "heat_transfer_coefficient", "boundary_property", 1),
            make_var("L", "length scale", "m", length, "length", "geometric_scale", 1),
            make_var("k", "thermal conductivity", "kg m s^-3 K^-1", thermal_conductivity, "thermal_conductivity", "material_property", -1),
            make_var("rho", "density", "kg m^-3", density, "density", "unused_context", 0),
        ],
    )

    # 9. Fourier number: Fo = alpha t / L^2
    add_sample(
        "dimless_0009",
        "Fourier number",
        "heat_transfer",
        "ratio of diffusive transport time to storage time",
        "alpha * t / L^2",
        [
            make_var("a", "thermal diffusivity", "m^2 s^-1", diffusivity, "diffusivity", "material_property", 1),
            make_var("t", "time", "s", time_dim, "time", "time_scale", 1),
            make_var("L", "length scale", "m", length, "length", "geometric_scale", -2),
            make_var("T0", "temperature", "K", temp_dim, "temperature", "unused_context", 0),
        ],
    )

    # 10. Bond number: Bo = delta_rho g L^2 / sigma
    add_sample(
        "dimless_0010",
        "Bond number",
        "fluid_mechanics",
        "ratio of gravitational to surface tension effects",
        "delta_rho * g * L^2 / sigma",
        [
            make_var("drho", "density difference", "kg m^-3", density, "density_difference", "material_contrast", 1),
            make_var("g", "gravitational acceleration", "m s^-2", gravity, "acceleration", "body_force_scale", 1),
            make_var("L", "length scale", "m", length, "length", "geometric_scale", 2),
            make_var("sigma", "surface tension", "kg s^-2", surface_tension, "surface_tension", "interface_property", -1),
            make_var("mu", "dynamic viscosity", "kg m^-1 s^-1", dyn_viscosity, "viscosity", "unused_context", 0),
        ],
    )

    # 11. Capillary number: Ca = mu v / sigma
    add_sample(
        "dimless_0011",
        "Capillary number",
        "fluid_mechanics",
        "ratio of viscous to surface tension effects",
        "mu * v / sigma",
        [
            make_var("mu", "dynamic viscosity", "kg m^-1 s^-1", dyn_viscosity, "viscosity", "material_property", 1),
            make_var("v", "velocity", "m s^-1", velocity, "velocity", "flow_scale", 1),
            make_var("sigma", "surface tension", "kg s^-2", surface_tension, "surface_tension", "interface_property", -1),
            make_var("rho", "density", "kg m^-3", density, "density", "unused_context", 0),
        ],
    )

    # 12. Cauchy number: Cy = rho v^2 / E
    add_sample(
        "dimless_0012",
        "Cauchy number",
        "solid_mechanics",
        "ratio of inertial to elastic effects",
        "rho * v^2 / E",
        [
            make_var("rho", "density", "kg m^-3", density, "density", "material_property", 1),
            make_var("v", "velocity", "m s^-1", velocity, "velocity", "motion_scale", 2),
            make_var("E", "elastic modulus", "kg m^-1 s^-2", pressure, "elastic_modulus", "material_property", -1),
            make_var("L", "length scale", "m", length, "length", "unused_context", 0),
        ],
    )

    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Data] Wrote {len(samples)} samples to {path}")
    return samples


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prior_knowledge_database(
    path="prior_knowledge_database",
    selected_training_case=training_case,
):
    """
    Load the per-entry JSON files exported from dpq.db.

    Expected JSON shape:
        {
          "source_id": "3.3.2",
          "name": "Bingham Compression Number",
          "variables": [
            {
              "name": "minimal yield strength",
              "unit": "Pa",
              "dimension_vector": {"M": 1, "L": -1, ...},
              "alpha": 1
            }
          ],
          "source": {...}
        }

    The learning code was originally written for add_sample/make_var records.
    This adapter fills the missing fields with empty strings and keeps alpha as
    the supervised target.
    """
    path = Path(path)
    samples = []
    for json_path in sorted(path.glob("*.json"), key=lambda p: p.stem):
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        sample = normalize_prior_sample(raw)
        sample["_json_path"] = str(json_path)
        samples.append(sample)

    case_key = str(selected_training_case or "").strip().lower().replace("-", "_")

    if case_key == "benard":
        excluded_names = {
            "Rayleigh Number",
            "Rayleigh Number (Heat Instability)",
            "Prandtl Number",
            "Nusselt Number",
        }
    elif case_key == "colebrook":
        excluded_names = {
            "Reynolds Number",
            "Relative Pipe Roughness",
            "Relative Roughness",
            "Friction Factor",
            "Moody Friction Factor",
        }
    elif case_key == "rayleigh":
        excluded_names = {
            "Fourier Hydrodynamic Number (Fourier Flow Number)",
            "Reynolds Number",
            "Strouhal Number",
        }
    elif case_key == "mhd":
        excluded_names = {
            "Hagen Number",
            "Hartmann Number",
            "Reynolds Number",
        }
    elif case_key == "blasius":
        excluded_names = {
            "Blasius Shape of Boundary Layer",
            "Reynolds Number",
        }
    elif case_key == "lid_cavity":
        excluded_names = {
            "Reynolds Number",
        }
    elif case_key == "rad":
        excluded_names = {
            "Pe´clet Difference Number",
            "Pe´clet Grid Number",
            "Pe´clet Mass and Vortex Number",
            "Pe´clet Mass Number",
            "Damko˝ hler Numerical Number",
            "Damko˝ hler Number",
            "Damko˝ hler Number Hydrodynamic",
            "Damko˝ hler Number Diffusion",
        }
    else:
        excluded_names = set()

    sample_count_before = len(samples)
    if excluded_names:
        samples = [sample for sample in samples if sample.get("name") not in excluded_names]
    print(
        "[Data] training_case={} | excluded prior samples={}".format(
            case_key or "global",
            sample_count_before - len(samples),
        )
    )

    extend_alpha_values_from_samples(samples)
    return samples


def normalize_prior_sample(raw):
    source = raw.get("source") or {}
    variables = []
    for i, var in enumerate(raw.get("variables") or []):
        name = var.get("name") or f"var_{i + 1}"
        variables.append(
            {
                "symbol": var.get("symbol") or symbol_from_name(name, i),
                "name": name,
                "unit": var.get("unit") or "",
                "dimension_vector": normalize_dimension_vector(var.get("dimension_vector")),
                "quantity_type": var.get("quantity_type") or "",
                "role": var.get("role") or "",
                "alpha": normalize_alpha_value(var.get("alpha")),
            }
        )

    return {
        "id": raw.get("source_id") or raw.get("id") or "",
        "name": raw.get("name") or "",
        "domain": raw.get("domain") or source.get("section_id") or "",
        "physical_meaning": raw.get("description") or "",
        "formula": raw.get("formula") or "unknown",
        "variables": variables,
        "source": source,
    }


def normalize_dimension_vector(value):
    if value is None:
        return None
    return {base: normalize_numeric_value(value.get(base, 0), default=0) for base in BASE_DIMS}


def symbol_from_name(name, index):
    cleaned = re_sub_non_identifier(name).strip("_")
    return cleaned or f"x{index + 1}"


def re_sub_non_identifier(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def has_trainable_labels(sample):
    if not sample.get("variables"):
        return False
    for var in sample["variables"]:
        if var.get("dimension_vector") is None:
            return False
        alpha = normalize_alpha_value(var.get("alpha"))
        if alpha is None or alpha not in ALPHA_TO_IDX:
            return False
    return True


def load_prior_knowledge_case(case_id, path="prior_knowledge_database"):
    """
    Load one case from prior_knowledge_database by source_id/stem/path.

    Examples:
        load_prior_knowledge_case("4.4.25")
        load_prior_knowledge_case("4.4.25.json")
        load_prior_knowledge_case("prior_knowledge_database/4.4.25.json")
    """
    root = Path(path)
    case_text = str(case_id)
    direct_path = Path(case_text)

    if direct_path.exists():
        json_path = direct_path
    else:
        name = case_text if case_text.endswith(".json") else f"{case_text}.json"
        json_path = root / name

    if not json_path.exists():
        for candidate in sorted(root.glob("*.json"), key=lambda p: p.stem):
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            if raw.get("source_id") == case_text or raw.get("id") == case_text:
                json_path = candidate
                break
        else:
            raise FileNotFoundError(f"Cannot find prior knowledge case: {case_id}")

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    sample = normalize_prior_sample(raw)
    sample["_json_path"] = str(json_path)
    return sample


def inspect_dimension_check(case_id, path="prior_knowledge_database", show_zero_terms=True):
    """
    Print the dimensional-balance process for one prior-knowledge case.

    For every variable and base dimension, this shows:
        contribution = dimension_exponent * alpha

    A dimensionless case should have zero total contribution for every base
    dimension in BASE_DIMS.
    """
    sample = load_prior_knowledge_case(case_id, path=path)

    print("\n" + "=" * 80)
    print(f"Dimension check for [{sample.get('id', '')}] {sample.get('name', '')}")
    print(f"JSON path: {sample.get('_json_path', '')}")
    print(f"Formula: {sample.get('formula', '')}")
    print("=" * 80)

    print("\nVariables")
    for i, var in enumerate(sample.get("variables", []), start=1):
        alpha = var.get("alpha")
        dim_vec = var.get("dimension_vector") or {}
        dim_text = ", ".join(
            f"{base}:{format_numeric_value(dim_vec.get(base, 0))}"
            for base in BASE_DIMS
        )
        print(
            f"  {i}. {var.get('symbol', '')} | {var.get('name', '')} | "
            f"unit={var.get('unit', '')} | alpha={format_alpha_value(alpha)}"
        )
        print(f"     dim: {dim_text}")

    print("\nPer-base contributions: dimension_exponent * alpha")
    residuals = {}
    all_zero = True

    for base in BASE_DIMS:
        terms = []
        total = Fraction(0, 1)

        for var in sample.get("variables", []):
            dim_vec = var.get("dimension_vector") or {}
            dim_power = numeric_to_fraction(dim_vec.get(base, 0), default=0)
            alpha = numeric_to_fraction(var.get("alpha"), default=0)
            contribution = dim_power * alpha
            total += contribution

            if show_zero_terms or contribution != 0:
                symbol = var.get("symbol", "")
                terms.append(
                    f"{symbol}: {format_numeric_value(dim_power)}*"
                    f"{format_alpha_value(alpha)}={format_numeric_value(contribution)}"
                )

        residuals[base] = total
        if total != 0:
            all_zero = False

        term_text = " + ".join(terms) if terms else "0"
        status = "OK" if total == 0 else "NOT OK"
        print(f"  {base:>5}: {term_text}  =>  {format_numeric_value(total)} [{status}]")

    print("\nResult")
    if all_zero:
        print("  Dimensionless: yes")
    else:
        print("  Dimensionless: no")

    return sample, residuals


def filter_trainable_samples(samples, require_dimensionless=True):
    usable = []
    skipped = []
    for sample in samples:
        reason = None
        if not has_trainable_labels(sample):
            reason = "missing dimension_vector or alpha outside ALPHA_VALUES"
        elif require_dimensionless and not is_dimensionless_sample(sample):
            residual = dimension_matrix(sample) @ alpha_vector(sample)
            reason = f"not dimensionless, residual={residual.tolist()}"

        if reason is None:
            usable.append(sample)
        else:
            skipped.append((sample, reason))
    return usable, skipped


# ============================================================
# 3. Dataset and feature encoding
# ============================================================

def build_vocabs(samples):
    """
    Keep the model input restricted to dimensional information only.

    The model should learn each variable's most likely exponent from the
    dimension vectors inside the current variable set, not from names, roles,
    domains, or other metadata.
    """
    return {
        "base_dims": list(BASE_DIMS),
        "input_dim": len(BASE_DIMS),
    }


def encode_variable(var, vocabs=None):
    dimension_vector = var.get("dimension_vector") or {}
    base_dims = vocabs.get("base_dims", BASE_DIMS) if vocabs else BASE_DIMS
    return [float(dimension_vector.get(b, 0)) for b in base_dims]


class DimensionlessDataset(Dataset):
    def __init__(self, samples, vocabs):
        self.samples = samples
        self.vocabs = vocabs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = []
        y = []

        for v in sample["variables"]:
            x.append(encode_variable(v, self.vocabs))
            alpha = normalize_alpha_value(v.get("alpha"), default=0)
            y.append(ALPHA_TO_IDX[alpha])

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
            "sample": sample,
        }


def collate_batch(batch):
    max_n = max(item["x"].shape[0] for item in batch)
    feat_dim = batch[0]["x"].shape[1]
    batch_size = len(batch)

    x_pad = torch.zeros(batch_size, max_n, feat_dim, dtype=torch.float32)
    y_pad = torch.full((batch_size, max_n), -100, dtype=torch.long)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)

    samples = []

    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        x_pad[i, :n] = item["x"]
        y_pad[i, :n] = item["y"]
        mask[i, :n] = True
        samples.append(item["sample"])

    return x_pad, y_pad, mask, samples


# ============================================================
# 4. Set Transformer model
# ============================================================

class SABlock(nn.Module):
    """
    Set Attention Block.
    This is a permutation-equivariant block suitable for variable sets.
    """
    def __init__(self, dim_hidden, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

        self.ffn = nn.Sequential(
            nn.Linear(dim_hidden, 4 * dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim_hidden, dim_hidden),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        key_padding_mask = ~mask  # True means padding for MultiheadAttention

        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        # 避免 padding 位置携带无意义信息
        x = x * mask.unsqueeze(-1)
        return x


class SetTransformerExponentClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        num_classes=None,
        dropout=0.1,
    ):
        super().__init__()
        if num_classes is None:
            num_classes = len(ALPHA_VALUES)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [SABlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask):
        h = self.input_proj(x)
        h = h * mask.unsqueeze(-1)

        for block in self.blocks:
            h = block(h, mask)

        logits = self.classifier(h)
        return logits


# ============================================================
# 5. Training
# ============================================================

def get_model_device(model):
    return next(model.parameters()).device


def alpha_values_signature():
    return [format_alpha_value(a) for a in ALPHA_VALUES]


def checkpoint_metadata(input_dim, vocabs, training_context=None):
    metadata = {
        "input_dim": input_dim,
        "num_classes": len(ALPHA_VALUES),
        "alpha_values": alpha_values_signature(),
        "base_dims": list(vocabs.get("base_dims", BASE_DIMS)),
        "model": "SetTransformerExponentClassifier",
    }
    if training_context is not None:
        metadata["training_context"] = training_context
    return metadata


def metadata_matches(saved, current):
    keys = [
        "input_dim",
        "num_classes",
        "alpha_values",
        "base_dims",
        "model",
        "training_context",
    ]
    return all(saved.get(key) == current.get(key) for key in keys)


def train_model(
    samples,
    vocabs,
    epochs=300,
    batch_size=4,
    lr=1e-3,
    device=None,
    force_retrain=False,
    checkpoint_dir="model_save",
    checkpoint_name="knowledge_aware_learning_best.pt",
    training_context=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"[Train] Using device: {device}")

    dataset = DimensionlessDataset(samples, vocabs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    input_dim = dataset[0]["x"].shape[1]
    metadata = checkpoint_metadata(
        input_dim,
        vocabs,
        training_context=training_context,
    )
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name

    model = SetTransformerExponentClassifier(input_dim=input_dim).to(device)

    if checkpoint_path.exists() and not force_retrain:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_metadata = checkpoint.get("metadata", {})
        if metadata_matches(saved_metadata, metadata):
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            checkpoint_loss = checkpoint.get("loss")
            loss_text = f"{checkpoint_loss:.4f}" if checkpoint_loss is not None else "unknown"
            print(
                f"[Train] Loaded checkpoint from {checkpoint_path} "
                f"(epoch={checkpoint.get('epoch')}, loss={loss_text})"
            )
            return model
        print(f"[Train] Checkpoint metadata mismatch. Retraining instead: {checkpoint_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_state_dict = None

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for x, y, mask, _ in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            logits = model(x, mask)
            loss = criterion(
                logits.reshape(-1, len(ALPHA_VALUES)),
                y.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "metadata": metadata,
                    "epoch": epoch,
                    "loss": best_loss,
                },
                checkpoint_path,
            )

        if epoch % 50 == 0 or epoch == 1:
            print(f"[Train] Epoch {epoch:04d} | loss = {avg_loss:.4f} | best = {best_loss:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    print(f"[Train] Best checkpoint saved to {checkpoint_path} | loss = {best_loss:.4f}")
    return model


# ============================================================
# 6. Prediction utilities
# ============================================================

@torch.no_grad()
def predict_distributions(model, sample, vocabs):
    model.eval()
    device = get_model_device(model)

    x = []
    for v in sample["variables"]:
        x.append(encode_variable(v, vocabs))

    x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)

    logits = model(x, mask)[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    return probs


def print_rolling_distributions(model, sample, vocabs, title="Prediction"):
    probs = predict_distributions(model, sample, vocabs)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, v in enumerate(sample["variables"]):
        dist = probs[i]
        order = np.argsort(-dist)

        dim_vec = v.get("dimension_vector") or {}
        dim_text = ", ".join(
            f"{base}:{format_numeric_value(dim_vec.get(base, 0))}"
            for base in BASE_DIMS
        )
        print(f"\nVariable {i + 1}: {v['symbol']} | unit = {v['unit']} | dim = [{dim_text}]")
        if "alpha" in v:
            print(f"  true alpha = {format_alpha_value(v['alpha'])}")

        print("  predicted exponent distribution:")
        for idx in order:
            alpha = IDX_TO_ALPHA[idx]
            print(f"    alpha = {format_alpha_value(alpha):>4}: p = {dist[idx]:.4f}")


def canonical_alpha(alpha):
    """
    Reduce equivalent powers. For example:
    [2, 2, 2, -2] -> [1, 1, 1, -1]
    """
    fractions = [normalize_alpha_value(a, default=0) for a in alpha]

    if all(a == 0 for a in fractions):
        return tuple(0 for _ in fractions)

    denom_lcm = 1
    for a in fractions:
        frac = a if isinstance(a, Fraction) else Fraction(a, 1)
        denom_lcm = math.lcm(denom_lcm, frac.denominator)

    arr = []
    for a in fractions:
        frac = a if isinstance(a, Fraction) else Fraction(a, 1)
        arr.append(int(frac * denom_lcm))

    nonzero = [abs(x) for x in arr if x != 0]
    g = int(reduce(math.gcd, nonzero))
    arr = [x // g for x in arr]

    # Fix sign so the first nonzero exponent is positive.
    first = next(i for i, value in enumerate(arr) if value != 0)
    if arr[first] < 0:
        arr = [-x for x in arr]

    return tuple(arr)


def alpha_to_formula(sample, alpha):
    positive_terms = []
    negative_terms = []

    for v, a in zip(sample["variables"], alpha):
        a = normalize_alpha_value(a, default=0)
        if a == 0:
            continue

        symbol = v["symbol"]
        abs_a = abs(a)

        if abs_a == 1:
            term = symbol
        else:
            term = f"{symbol}^{format_alpha_value(abs_a)}"

        if a > 0:
            positive_terms.append(term)
        else:
            negative_terms.append(term)

    numerator = " * ".join(positive_terms) if positive_terms else "1"
    denominator = " * ".join(negative_terms)

    if denominator:
        return f"{numerator} / ({denominator})"
    return numerator


def make_subset_sample(sample, indices):
    subset = dict(sample)
    subset["variables"] = [sample["variables"][i] for i in indices]
    subset["_parent_id"] = sample.get("id", "")
    subset["_subset_indices"] = tuple(indices)
    return subset


def alpha_to_full_alpha(sample, subset_indices, subset_alpha):
    full_alpha = [0] * len(sample["variables"])
    for idx, alpha in zip(subset_indices, subset_alpha):
        full_alpha[idx] = alpha
    return tuple(full_alpha)


def variable_symbols(sample):
    return [v.get("symbol", f"x{i + 1}") for i, v in enumerate(sample["variables"])]


def dimension_matrix_fractions(sample):
    variables = sample["variables"]
    return [
        [
            numeric_to_fraction((v.get("dimension_vector") or {}).get(base, 0), default=0)
            for v in variables
        ]
        for base in BASE_DIMS
    ]


def rref_fraction_matrix(matrix):
    mat = [[Fraction(value) for value in row] for row in matrix]
    if not mat:
        return mat, []

    row_count = len(mat)
    col_count = len(mat[0])
    pivot_cols = []
    row = 0

    for col in range(col_count):
        pivot = None
        for candidate in range(row, row_count):
            if mat[candidate][col] != 0:
                pivot = candidate
                break

        if pivot is None:
            continue

        mat[row], mat[pivot] = mat[pivot], mat[row]
        pivot_value = mat[row][col]
        mat[row] = [value / pivot_value for value in mat[row]]

        for other_row in range(row_count):
            if other_row == row:
                continue
            factor = mat[other_row][col]
            if factor == 0:
                continue
            mat[other_row] = [
                value - factor * pivot_part
                for value, pivot_part in zip(mat[other_row], mat[row])
            ]

        pivot_cols.append(col)
        row += 1
        if row == row_count:
            break

    return mat, pivot_cols


def buckingham_pi_nullspace_basis(sample):
    matrix = dimension_matrix_fractions(sample)
    if not matrix or not matrix[0]:
        return []

    rref, pivot_cols = rref_fraction_matrix(matrix)
    n = len(matrix[0])
    free_cols = [col for col in range(n) if col not in pivot_cols]
    basis = []

    for free_col in free_cols:
        vector = [Fraction(0, 1) for _ in range(n)]
        vector[free_col] = Fraction(1, 1)

        for pivot_row, pivot_col in enumerate(pivot_cols):
            vector[pivot_col] = -rref[pivot_row][free_col]

        if any(value != 0 for value in vector):
            basis.append(tuple(normalize_numeric_value(value) for value in vector))

    return basis


def dimension_rank(sample):
    matrix = dimension_matrix_fractions(sample)
    if not matrix or not matrix[0]:
        return 0
    _, pivot_cols = rref_fraction_matrix(matrix)
    return len(pivot_cols)


def simplest_buckingham_pi_candidate(sample, complexity_penalty=0.05):
    basis = buckingham_pi_nullspace_basis(sample)
    if not basis:
        return None

    candidates = []
    for raw_alpha in basis:
        canonical = canonical_alpha(raw_alpha)
        if all(value == 0 for value in canonical):
            continue

        alpha_arr = np.array(canonical, dtype=float)
        complexity = np.sum(np.abs(alpha_arr)) + 0.1 * np.count_nonzero(alpha_arr)
        candidates.append(
            {
                "score": -complexity_penalty * complexity,
                "raw_alpha": raw_alpha,
                "canonical_alpha": canonical,
                "source": "buckingham_pi",
            }
        )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda cand: (
            np.sum(np.abs(np.array(cand["canonical_alpha"], dtype=float))),
            np.count_nonzero(np.array(cand["canonical_alpha"], dtype=float)),
        ),
    )


def constrained_decode(
    model,
    sample,
    vocabs,
    top_k_per_variable=5,
    top_m_candidates=10,
    complexity_penalty=0.05,
    verbose=True,
    probs=None,
):
    """
    Decode alpha under the hard dimensional constraint D alpha = 0.
    The model provides a prior distribution, but the final candidates must be dimensionless.
    """
    if probs is None:
        probs = predict_distributions(model, sample, vocabs)
    D = dimension_matrix(sample)
    n = len(sample["variables"])

    # For each variable, retain top-k exponent values.
    choices = []
    for i in range(n):
        order = np.argsort(-probs[i])
        selected = [IDX_TO_ALPHA[idx] for idx in order[:top_k_per_variable]]
        choices.append(selected)

    candidates = {}
    log_probs = np.log(probs + 1e-12)

    def try_add(alpha_tuple):
        alpha_arr = np.array(alpha_tuple, dtype=float)

        if np.allclose(alpha_arr, 0.0):
            return

        if not np.allclose(D @ alpha_arr, 0.0, atol=1e-8):
            return

        prior_score = 0.0
        for i, a in enumerate(alpha_tuple):
            prior_score += log_probs[i, ALPHA_TO_IDX[normalize_alpha_value(a)]]

        complexity = np.sum(np.abs(alpha_arr)) + 0.1 * np.count_nonzero(alpha_arr)
        score = prior_score - complexity_penalty * complexity

        key = canonical_alpha(alpha_tuple)

        if key not in candidates or score > candidates[key]["score"]:
            candidates[key] = {
                "score": score,
                "raw_alpha": tuple(normalize_alpha_value(x) for x in alpha_tuple),
                "canonical_alpha": key,
            }

    # First search in the high-probability subset.
    for alpha_tuple in itertools.product(*choices):
        try_add(alpha_tuple)

    # If no valid answer is found, fall back to the simplest Buckingham-pi
    # nullspace vector instead of enumerating the full exponent space.
    if len(candidates) == 0:
        if verbose:
            print("[Decode] No valid candidate in top-k space. Using Buckingham-pi fallback...")
        buckingham_candidate = simplest_buckingham_pi_candidate(sample, complexity_penalty)
        if buckingham_candidate is not None:
            candidates[buckingham_candidate["canonical_alpha"]] = buckingham_candidate

    ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_m_candidates]


def model_log_prob_for_alpha(probs, alpha):
    score = 0.0
    for i, value in enumerate(alpha):
        value = normalize_alpha_value(value)
        if value not in ALPHA_TO_IDX:
            return None
        score += np.log(probs[i, ALPHA_TO_IDX[value]] + 1e-12)
    return float(score)


def alpha_complexity(alpha):
    alpha_arr = np.array(alpha, dtype=float)
    return float(np.sum(np.abs(alpha_arr)) + 0.1 * np.count_nonzero(alpha_arr))


def evaluate_subset_candidates(
    model,
    sample,
    vocabs,
    min_subset_size=2,
    max_subset_size=None,
    top_k_per_variable=5,
    top_m_per_subset=3,
    top_n=3,
    subset_size_penalty=1.0,
    complexity_penalty=0.05,
    show_progress=False,
    progress_desc="Evaluating subsets",
):
    """
    Enumerate variable subsets and decode each subset independently.

    This is intended for inference with possible irrelevant variables. The model
    is rerun for each subset because the exponent distribution depends on the
    variable set context.
    """
    n = len(sample["variables"])
    if max_subset_size is None:
        max_subset_size = n
    max_subset_size = min(max_subset_size, n)
    candidates = {}
    total_subsets = sum(
        math.comb(n, k)
        for k in range(min_subset_size, max_subset_size + 1)
    )
    progress_bar = None
    processed_subsets = 0
    next_progress_report = 0.0

    if show_progress:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_subsets, desc=progress_desc, unit="subset")
        except ImportError:
            print(f"[Progress] {progress_desc}: 0/{total_subsets} subsets")

    for k in range(min_subset_size, max_subset_size + 1):
        for indices in itertools.combinations(range(n), k):
            processed_subsets += 1
            subset = make_subset_sample(sample, indices)
            if dimension_rank(subset) == k:
                if progress_bar is not None:
                    progress_bar.update(1)
                elif show_progress:
                    fraction = processed_subsets / max(total_subsets, 1)
                    if fraction >= next_progress_report or processed_subsets == total_subsets:
                        print(
                            f"[Progress] {progress_desc}: "
                            f"{processed_subsets}/{total_subsets} subsets "
                            f"({100 * fraction:.1f}%)"
                        )
                        next_progress_report += 0.1
                continue

            probs = predict_distributions(model, subset, vocabs)
            decoded = constrained_decode(
                model,
                subset,
                vocabs,
                top_k_per_variable=top_k_per_variable,
                top_m_candidates=top_m_per_subset,
                complexity_penalty=complexity_penalty,
                verbose=False,
                probs=probs,
            )

            for cand in decoded:
                subset_alpha = cand["canonical_alpha"]
                if len(subset_alpha) != k:
                    continue

                score_alpha = cand.get("raw_alpha", subset_alpha)
                log_prob = model_log_prob_for_alpha(probs, score_alpha)
                if log_prob is None:
                    log_prob = cand["score"]

                complexity = alpha_complexity(subset_alpha)
                size_penalty = subset_size_penalty * k
                alpha_penalty = complexity_penalty * complexity
                score = log_prob - size_penalty - alpha_penalty
                full_alpha = alpha_to_full_alpha(sample, indices, subset_alpha)
                key = (indices, subset_alpha)

                item = {
                    "score": score,
                    "model_log_prob": log_prob,
                    "complexity": complexity,
                    "size_penalty": size_penalty,
                    "alpha_penalty": alpha_penalty,
                    "subset_size_penalty": subset_size_penalty,
                    "complexity_penalty": complexity_penalty,
                    "subset_size": k,
                    "subset_indices": tuple(indices),
                    "subset_sample": subset,
                    "subset_alpha": subset_alpha,
                    "full_alpha": full_alpha,
                    "formula": alpha_to_formula(subset, subset_alpha),
                    "source": cand.get("source", "model_subset"),
                    "raw_alpha": cand.get("raw_alpha", subset_alpha),
                }

                if key not in candidates or score > candidates[key]["score"]:
                    candidates[key] = item

            if progress_bar is not None:
                progress_bar.update(1)
            elif show_progress:
                fraction = processed_subsets / max(total_subsets, 1)
                if fraction >= next_progress_report or processed_subsets == total_subsets:
                    print(
                        f"[Progress] {progress_desc}: "
                        f"{processed_subsets}/{total_subsets} subsets "
                        f"({100 * fraction:.1f}%)"
                    )
                    next_progress_report += 0.1

    if progress_bar is not None:
        progress_bar.close()
    ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]


def print_subset_candidates(
    model,
    sample,
    vocabs,
    title="Subset candidates",
    top_n=3,
    max_subset_size=None,
    subset_size_penalty=1.0,
):
    candidates = evaluate_subset_candidates(
        model,
        sample,
        vocabs,
        top_n=top_n,
        max_subset_size=max_subset_size,
        subset_size_penalty=subset_size_penalty,
    )

    print("\n" + "=" * 80)
    print(title + " | best dimensionless candidates across variable subsets")
    print("=" * 80)

    if not candidates:
        print("No valid subset candidate found.")
        return []

    symbols = variable_symbols(sample)
    for rank, cand in enumerate(candidates, start=1):
        subset_symbols = [symbols[i] for i in cand["subset_indices"]]
        residual = dimension_matrix(cand["subset_sample"]) @ np.array(
            cand["subset_alpha"], dtype=float
        )

        print(f"\nRank {rank}")
        print(f"  score           = {cand['score']:.4f}")
        print(f"  model log prob  = {cand['model_log_prob']:.4f}")
        print(f"  size penalty    = {cand['size_penalty']:.4f}")
        print(f"  alpha penalty   = {cand['alpha_penalty']:.4f}")
        print(f"  variables       = {subset_symbols}")
        print(f"  subset indices  = {cand['subset_indices']}")
        print(f"  raw alpha       = {cand['raw_alpha']}")
        print(f"  subset alpha    = {cand['subset_alpha']}")
        print(f"  full alpha      = {cand['full_alpha']}")
        print(f"  formula         = {cand['formula']}")
        print(f"  source          = {cand['source']}")
        print(f"  D_subset@alpha  = {residual}")

    return candidates


def print_specific_subset_debug(
    model,
    sample,
    vocabs,
    symbols,
    top_k_per_variable=5,
    top_m_candidates=5,
    subset_size_penalty=1.0,
    complexity_penalty=0.05,
    title="Specific subset debug",
):
    symbol_to_index = {
        var.get("symbol", f"x{i + 1}"): i
        for i, var in enumerate(sample["variables"])
    }
    missing = [symbol for symbol in symbols if symbol not in symbol_to_index]
    if missing:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"Missing symbols: {missing}")
        return []

    indices = tuple(symbol_to_index[symbol] for symbol in symbols)
    subset = make_subset_sample(sample, indices)
    probs = predict_distributions(model, subset, vocabs)

    print("\n" + "=" * 80)
    print(title + f" | symbols = {tuple(symbols)}")
    print("=" * 80)
    print(f"subset indices: {indices}")
    print(f"dimension rank: {dimension_rank(subset)} / {len(indices)}")

    print("\nVariable-wise predicted distributions")
    choices = []
    for i, var in enumerate(subset["variables"]):
        order = np.argsort(-probs[i])
        selected = [IDX_TO_ALPHA[idx] for idx in order[:top_k_per_variable]]
        choices.append(selected)

        dim_vec = var.get("dimension_vector") or {}
        dim_text = ", ".join(
            f"{base}:{format_numeric_value(dim_vec.get(base, 0))}"
            for base in BASE_DIMS
        )
        print(f"\n  {var.get('symbol', '')} | dim = [{dim_text}]")
        print(f"    top-{top_k_per_variable}: {[format_alpha_value(a) for a in selected]}")
        for idx in order:
            alpha = IDX_TO_ALPHA[idx]
            print(f"    alpha = {format_alpha_value(alpha):>4}: p = {probs[i, idx]:.4f}")

    print("\nTop-k search space")
    print("  " + " x ".join(str([format_alpha_value(a) for a in choice]) for choice in choices))

    candidates = constrained_decode(
        model,
        subset,
        vocabs,
        top_k_per_variable=top_k_per_variable,
        top_m_candidates=top_m_candidates,
        verbose=True,
        probs=probs,
    )

    print("\nDecoded candidates for this subset")
    if not candidates:
        print("  No valid candidate found.")
    for rank, cand in enumerate(candidates, start=1):
        alpha = cand["canonical_alpha"]
        residual = dimension_matrix(subset) @ np.array(alpha, dtype=float)
        log_prob = model_log_prob_for_alpha(probs, cand.get("raw_alpha", alpha))
        complexity = alpha_complexity(alpha)
        size_penalty = subset_size_penalty * len(indices)
        alpha_penalty = complexity_penalty * complexity
        subset_score = None
        if log_prob is not None:
            subset_score = log_prob - size_penalty - alpha_penalty
        source = cand.get("source", "model_subset")
        full_alpha = alpha_to_full_alpha(sample, indices, alpha)
        print(f"  Rank {rank}")
        print(f"    source       = {source}")
        print(f"    raw alpha    = {cand.get('raw_alpha', alpha)}")
        print(f"    subset alpha = {alpha}")
        print(f"    full alpha   = {full_alpha}")
        print(f"    log prob     = {log_prob:.4f}" if log_prob is not None else "    log prob     = unavailable")
        if subset_score is not None:
            print(f"    subset score = {subset_score:.4f}")
            print(f"    size penalty = {size_penalty:.4f}")
            print(f"    alpha penalty= {alpha_penalty:.4f}")
        print(f"    score        = {cand['score']:.4f}")
        print(f"    formula      = {alpha_to_formula(subset, alpha)}")
        print(f"    D@alpha      = {residual}")

    print("\nBuckingham-pi nullspace basis")
    basis = buckingham_pi_nullspace_basis(subset)
    if not basis:
        print("  No non-trivial nullspace basis.")
    for vec in basis:
        canonical = canonical_alpha(vec)
        residual = dimension_matrix(subset) @ np.array(canonical, dtype=float)
        print(f"  raw={vec}, canonical={canonical}, formula={alpha_to_formula(subset, canonical)}, D@alpha={residual}")

    return candidates


def print_constrained_candidates(model, sample, vocabs, title="Constrained candidates"):
    print_rolling_distributions(model, sample, vocabs, title=title + " | variable-wise distributions")
    return print_subset_candidates(
        model,
        sample,
        vocabs,
        title=title,
        top_n=3,
    )


def evaluate_test_samples(model, test_samples, vocabs, max_print=12):
    print("\n" + "=" * 80)
    print("Evaluation on original add_sample toy dimensionless numbers")
    print("=" * 80)

    correct = 0
    evaluated = 0

    for sample in test_samples[:max_print]:
        true_alpha = canonical_alpha([v.get("alpha", 0) for v in sample["variables"]])
        candidates = evaluate_subset_candidates(
            model,
            sample,
            vocabs,
            top_n=3,
            subset_size_penalty=1.0,
        )
        top_alpha = candidates[0]["full_alpha"] if candidates else None
        is_correct = top_alpha == true_alpha
        correct += int(is_correct)
        evaluated += 1

        print(f"\n[{sample['id']}] {sample['name']}")
        print(f"  true formula    = {sample.get('formula', '')}")
        print(f"  true alpha      = {true_alpha}")
        if top_alpha is None:
            print("  predicted       = no valid dimensionless candidate")
            continue
        print(f"  predicted alpha = {top_alpha}")
        print(f"  predicted form  = {candidates[0]['formula']}")
        print(f"  match           = {is_correct}")

        print("  top subset candidates:")
        for rank, cand in enumerate(candidates, start=1):
            subset_symbols = [
                sample["variables"][i].get("symbol", f"x{i + 1}")
                for i in cand["subset_indices"]
            ]
            print(
                f"    {rank}. score={cand['score']:.4f}, "
                f"source={cand['source']}, "
                f"k={cand['subset_size']}, size_penalty={cand['size_penalty']:.2f}, "
                f"vars={subset_symbols}, alpha={cand['subset_alpha']}, "
                f"full={cand['full_alpha']}, formula={cand['formula']}"
            )

    if evaluated:
        print("\n" + "-" * 80)
        print(f"Top-1 exact canonical-alpha accuracy: {correct}/{evaluated} = {correct / evaluated:.3f}")


# ============================================================
# 7. New sample for prediction
# ============================================================

def build_new_sample():
    """
    A new variable set similar to Reynolds number, with an extra time variable.
    The model should ideally recover something close to rho * u * d / eta,
    with alpha for t being zero.
    """
    density = dim(M=1, L=-3)
    velocity = dim(L=1, T=-1)
    length = dim(L=1)
    dyn_viscosity = dim(M=1, L=-1, T=-1)
    time_dim = dim(T=1)

    sample = {
        "id": "new_problem_0001",
        "name": "New candidate variable set",
        "domain": "fluid_mechanics",
        "physical_meaning": "unknown dimensionless group to be discovered",
        "formula": "unknown",
        "variables": [
            make_var("rho_f", "fluid density", "kg m^-3", density, "density", "material_property"),
            make_var("u", "flow velocity", "m s^-1", velocity, "velocity", "flow_scale"),
            make_var("d", "particle diameter", "m", length, "length", "geometric_scale"),
            make_var("eta", "dynamic viscosity", "kg m^-1 s^-1", dyn_viscosity, "viscosity", "material_property"),
            make_var("t", "time", "s", time_dim, "time", "time_scale"),
        ],
    }

    return sample


# ============================================================
# 8. Main
# ============================================================

def main():
    set_seed(42)

    prior_path = "prior_knowledge_database"
    toy_test_path = "dimensionless_toy_database.jsonl"
    case_key = str(training_case or "").strip().lower().replace("-", "_")

    prior_samples = load_prior_knowledge_database(prior_path, training_case)
    train_samples, skipped_samples = filter_trainable_samples(prior_samples)

    if case_key in {"benard", "colebrook", "rayleigh", "mhd", "blasius","lid_cavity","rad"}:
        checkpoint_dir = Path("model_save") / "case_specific" / case_key
        checkpoint_name = f"knowledge_aware_learning_{case_key}_best.pt"
        training_context = {
            "case_name": case_key,
            "n_train_samples": len(train_samples),
        }
    else:
        checkpoint_dir = Path("model_save")
        checkpoint_name = "knowledge_aware_learning_best.pt"
        training_context = None

    print(f"[Data] Loaded {len(prior_samples)} prior JSON samples from {prior_path}")
    print(f"[Data] Alpha values: {[format_alpha_value(a) for a in ALPHA_VALUES]}")
    print(f"[Data] Trainable prior samples: {len(train_samples)}")
    print(f"[Case] Training case: {case_key or 'global'}")
    print(f"[Model] Checkpoint: {Path(checkpoint_dir) / checkpoint_name}")
    if skipped_samples:
        print(f"[Data] Skipped prior samples: {len(skipped_samples)}")
        for sample, reason in skipped_samples[:10]:
            print(f"  - {sample.get('id', '')} {sample.get('name', '')}: {reason}")

    if not train_samples:
        raise RuntimeError("No trainable samples found in prior_knowledge_database.")
  
    test_samples = build_toy_database(toy_test_path)

    # Feature configuration is dimension-only. Metadata such as quantity_type,
    # role, and domain is intentionally excluded from model inputs.
    vocabs = build_vocabs(train_samples + test_samples)

    print("\n[Features]")
    print("base_dims:", vocabs["base_dims"])
    print("input_dim:", vocabs["input_dim"])
    raise OSError
    model = train_model(
        samples=train_samples,
        vocabs=vocabs,
        epochs=3000,
        batch_size=400,
        lr=1e-3,
        force_retrain=False,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        training_context=training_context,
    )

    # 1. Validation on one prior-knowledge training sample
    train_sample = train_samples[10]
    print_rolling_distributions(
        model,
        train_sample,
        vocabs,
        title="Validation on one prior-knowledge training sample",
    )

    # 2. Test on the original add_sample toy dimensionless numbers
    evaluate_test_samples(model, test_samples, vocabs)



if __name__ == "__main__":
    main()
