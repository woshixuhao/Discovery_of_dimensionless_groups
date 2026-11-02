"""Local dimension-only model and inference utilities for the RAD checkpoint.

Training and prior-database loading are intentionally not part of this module.
"""

import math
from fractions import Fraction
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE_DIMS = ["M", "L", "T", "I", "Theta", "N", "J"]
ALPHA_VALUES = [-3, -2, -1, 0, 1, 2, 3]
ALPHA_TO_IDX = {a: i for i, a in enumerate(ALPHA_VALUES)}
IDX_TO_ALPHA = {i: a for a, i in ALPHA_TO_IDX.items()}


def load_checkpoint_model(path, device=None):
    """Restore the model and exponent order directly from checkpoint metadata."""
    global BASE_DIMS, ALPHA_VALUES, ALPHA_TO_IDX, IDX_TO_ALPHA

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("RAD knowledge model not found: {}".format(path))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("metadata"), dict):
        raise ValueError("Checkpoint must be a dictionary containing inference metadata.")
    metadata = checkpoint.get("metadata", {})
    state = checkpoint.get("model_state_dict")
    required = {"model", "base_dims", "input_dim", "alpha_values", "num_classes"}
    missing = sorted(required - metadata.keys())
    if missing or not isinstance(state, dict):
        raise ValueError("Checkpoint must contain model weights and inference metadata; missing: {}".format(missing))
    if metadata["model"] != "SetTransformerExponentClassifier":
        raise ValueError("Unsupported checkpoint model: {}".format(metadata["model"]))

    base_dims = list(metadata["base_dims"])
    if len(base_dims) != 7 or set(base_dims) != {"M", "L", "T", "I", "Theta", "N", "J"}:
        raise ValueError("Checkpoint must define the seven distinct SI base dimensions.")
    alpha_values = [normalize_alpha_value(value) for value in metadata["alpha_values"]]
    if not alpha_values or None in alpha_values or len(set(alpha_values)) != len(alpha_values):
        raise ValueError("Checkpoint exponent classes must be nonempty, valid and distinct.")
    if int(metadata["input_dim"]) != len(base_dims) or int(metadata["num_classes"]) != len(alpha_values):
        raise ValueError("Checkpoint feature/class counts do not match its metadata.")
    try:
        hidden_dim, input_dim = state["input_proj.weight"].shape
        class_count = state["classifier.2.weight"].shape[0]
        block_indices = {int(key.split(".")[1]) for key in state if key.startswith("blocks.")}
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("Checkpoint is missing the expected Set Transformer architecture.") from exc
    if input_dim != len(base_dims) or class_count != len(alpha_values):
        raise ValueError("Checkpoint weight shapes disagree with its feature/class metadata.")
    if not block_indices or block_indices != set(range(len(block_indices))):
        raise ValueError("Checkpoint attention blocks must have consecutive indices.")
    num_heads = int(metadata.get("num_heads", 4))
    if num_heads <= 0 or hidden_dim % num_heads:
        raise ValueError("Checkpoint hidden dimension is incompatible with its attention heads.")

    BASE_DIMS = base_dims
    ALPHA_VALUES = alpha_values
    ALPHA_TO_IDX = {value: i for i, value in enumerate(alpha_values)}
    IDX_TO_ALPHA = {i: value for i, value in enumerate(alpha_values)}
    model = SetTransformerExponentClassifier(
        input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
        num_layers=len(block_indices), num_classes=len(alpha_values),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    vocabs = {"base_dims": list(base_dims), "input_dim": input_dim}
    print("[Model] Loaded local checkpoint: {} (epoch={}, classes={}, device={})".format(
        path, checkpoint.get("epoch", "unknown"), len(alpha_values), device,
    ))
    return model, vocabs


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


def encode_variable(var, vocabs=None):
    dimension_vector = var.get("dimension_vector") or {}
    base_dims = vocabs.get("base_dims", BASE_DIMS) if vocabs else BASE_DIMS
    return [float(dimension_vector.get(b, 0)) for b in base_dims]


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


def get_model_device(model):
    return next(model.parameters()).device


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


def dimension_rank(sample):
    matrix = dimension_matrix_fractions(sample)
    if not matrix or not matrix[0]:
        return 0
    _, pivot_cols = rref_fraction_matrix(matrix)
    return len(pivot_cols)


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
