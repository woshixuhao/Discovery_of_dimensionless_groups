import argparse
import copy
import heapq
import itertools
import json
import pickle
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    from contextlib import redirect_stdout
except ImportError:
    class redirect_stdout:
        def __init__(self, new_target):
            self.new_target = new_target
            self.old_target = None

        def __enter__(self):
            self.old_target = sys.stdout
            sys.stdout = self.new_target
            return self.new_target

        def __exit__(self, exc_type, exc_value, traceback):
            sys.stdout = self.old_target

from knowledge_aware_learning import (
    alpha_to_formula,
    evaluate_subset_candidates,
)
from case_workflow import (
    find_example_sample,
    load_case_data,
    load_example_samples,
    load_prediction_model,
    load_processed_case,
    print_case_overview,
)


def load_settings(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_fraction_list(value, default):
    values = default if value is None else value
    if isinstance(values, str):
        values = [x.strip() for x in values.split(",") if x.strip()]
    return [Fraction(str(item)) for item in values]


def progress_iter(iterable, total=None, desc="Progress", enabled=False):
    if not enabled:
        for item in iterable:
            yield item
        return

    try:
        from tqdm import tqdm
        for item in tqdm(iterable, total=total, desc=desc, unit="item"):
            yield item
        return
    except ImportError:
        pass

    total = total or 0
    processed = 0
    next_report = 0.0
    print("[Progress] {0}: 0/{1}".format(desc, total if total else "?"))
    for item in iterable:
        processed += 1
        yield item
        if total:
            fraction = processed / max(total, 1)
            if fraction >= next_report or processed == total:
                print(
                    "[Progress] {0}: {1}/{2} ({3:.1f}%)".format(
                        desc,
                        processed,
                        total,
                        100 * fraction,
                    )
                )
                next_report += 0.1
        elif processed % 1000 == 0:
            print("[Progress] {0}: {1} items".format(desc, processed))


def parse_args():
    settings_parser = argparse.ArgumentParser(add_help=False)
    settings_parser.add_argument("--settings", default="settings.json")
    settings_args, remaining = settings_parser.parse_known_args()
    settings = load_settings(settings_args.settings)

    parser = argparse.ArgumentParser(
        description="Optimize a total dimensionless number from model-predicted Top-K pi groups."
    )
    parser.add_argument("--settings", default=settings_args.settings)
    parser.add_argument("case", nargs="?", default=settings.get("case", "benard"))
    parser.add_argument("--samples-jsonl", default=settings.get("samples_jsonl", "example_samples.jsonl"))
    parser.add_argument("--data-dir", default=settings.get("data_dir", "example_processed_data"))
    parser.add_argument("--examples-root", default=settings.get("examples_root", "Examples"))
    parser.add_argument("--prior-dir", default=settings.get("prior_dir", "prior_knowledge_database"))
    parser.add_argument("--checkpoint-dir", default=settings.get("checkpoint_dir", "model_save"))
    parser.add_argument(
        "--checkpoint-name",
        default=settings.get("checkpoint_name", "knowledge_aware_learning_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=settings.get("epochs", 3000))
    parser.add_argument("--batch-size", type=int, default=settings.get("batch_size", 400))
    parser.add_argument("--lr", type=float, default=settings.get("lr", 1e-3))
    parser.add_argument("--device", default=settings.get("device"))
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        default=bool(settings.get("force_retrain", False)),
    )
    parser.add_argument(
        "--no-cache-generated-data",
        action="store_true",
        default=bool(settings.get("no_cache_generated_data", False)),
    )
    parser.add_argument("--top-k", type=int, default=settings.get("parameter_top_k", settings.get("top_n", 5)))
    parser.add_argument("--independent-rank-top-n", type=int, default=settings.get("independent_rank_top_n", 10))
    parser.add_argument("--rank-source-top-k", type=int, default=settings.get("rank_source_top_k", None))
    parser.add_argument(
        "--classic-exponents",
        default=settings.get("classic_exponents", [-3, -2, -1, "-1/2", 0, "1/2", 1, 2, 3]),
    )
    parser.add_argument("--r2-threshold", type=float, default=settings.get("r2_threshold", 0.95))
    parser.add_argument("--polynomial-max-degree", type=int, default=settings.get("polynomial_max_degree", 5))
    parser.add_argument("--search-refine-top-m", type=int, default=settings.get("search_refine_top_m", 0))
    parser.add_argument(
        "--accelerated-min-rank",
        type=int,
        default=settings.get("accelerated_min_rank", 4),
        help="Use accelerated beta search when rank(A) is at least this value.",
    )
    parser.add_argument(
        "--accelerated-refine-top-m",
        type=int,
        default=settings.get("accelerated_refine_top_m", 5000),
        help="When accelerated search is active and search_refine_top_m <= 0, refine only this many Step-2 beta candidates.",
    )
    parser.add_argument(
        "--accelerated-search-chunk-size",
        type=int,
        default=settings.get("accelerated_search_chunk_size", 50000),
    )
    parser.add_argument(
        "--classic-max-active",
        type=int,
        default=settings.get("classic_max_active", 3),
        help="Maximum nonzero beta entries for accelerated classic exponent search. Use 0 to disable this limit.",
    )
    parser.add_argument(
        "--branch-bound-margin",
        type=float,
        default=settings.get("branch_bound_margin", 1e-12),
    )
    parser.add_argument(
        "--two-pi-exhaustive-search",
        action="store_true",
        default=bool(settings.get("two_pi_exhaustive_search", False)),
    )
    parser.add_argument(
        "--two-pi-complexity-weight",
        type=float,
        default=settings.get("two_pi_complexity_weight", 0.01),
    )
    parser.add_argument(
        "--force-retain",
        "--fore-retain",
        dest="force_retain",
        action="store_true",
        default=bool(settings.get("force_retain", settings.get("fore_retain", False))),
        help="Force recomputing raw_candidates instead of loading the cached file.",
    )
    parser.add_argument(
        "--skip-top-k-processing",
        action="store_true",
        default=bool(settings.get("skip_top_k_processing", False)),
        help="Use all Top-K pi groups without variance/correlation/rank pruning.",
    )
    parser.add_argument("--subset-size-penalty", type=float, default=settings.get("subset_size_penalty", 1.0))
    parser.add_argument("--output-index", type=int, default=settings.get("output_index", 0))
    parser.add_argument("--target-transform", default=settings.get("target_transform", "auto_log"))
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=settings.get("pi_variance_threshold", 1e-10),
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=settings.get("pi_corr_threshold", 0.995),
    )
    parser.add_argument("--cv", type=int, default=settings.get("elasticnet_cv", 5))
    parser.add_argument("--n-alphas", type=int, default=settings.get("elasticnet_n_alphas", 100))
    parser.add_argument(
        "--l1-ratios",
        default=settings.get("elasticnet_l1_ratios", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
    )
    parser.add_argument("--path-alphas", type=int, default=settings.get("lasso_path_alphas", 80))
    parser.add_argument(
        "--r2-drop-tolerance",
        type=float,
        default=settings.get("r2_drop_tolerance", 0.01),
    )
    parser.add_argument(
        "--fraction-max-denominator",
        type=int,
        default=settings.get("fraction_max_denominator", 6),
    )
    parser.add_argument(
        "--residual-corr-threshold",
        type=float,
        default=settings.get("residual_corr_threshold", 0.3),
    )
    return parser.parse_args(remaining)


def load_case_from_settings(args):
    sample, case_data = load_processed_case(args.data_dir, args.case)
    if sample is not None:
        return apply_case_specific_variable_filter(args, sample, case_data)

    samples = load_example_samples(args.samples_jsonl)
    sample = find_example_sample(samples, args.case)
    case_data = load_case_data(
        args.case,
        sample,
        args.data_dir,
        args.examples_root,
        cache_generated=not args.no_cache_generated_data,
    )
    return apply_case_specific_variable_filter(args, sample, case_data)


def canonical_case_name(case_name):
    return str(case_name).strip().strip("\"'").lower()


def safe_path_component(value):
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in str(value).strip()
    ) or "unknown"


def output_column_label(case_data, output_index):
    metadata = case_data.get("metadata") or {}
    output_columns = metadata.get("output_columns") or []
    if 0 <= output_index < len(output_columns):
        column = output_columns[output_index]
        label = column.get("symbol") or column.get("name")
        if label:
            return str(label).strip()
    return "output_{0}".format(output_index)


def target_output_indices(args, case_data):
    y_raw = np.asarray(case_data["Y"])
    if (
        canonical_case_name(args.case) == "wall_model"
        and y_raw.ndim == 2
        and y_raw.shape[1] > 1
    ):
        return list(range(y_raw.shape[1]))
    return [int(args.output_index)]


def apply_case_specific_variable_filter(args, sample, case_data):
    sample = copy.deepcopy(sample)
    case_data = dict(case_data)

    if canonical_case_name(args.case) == "malkus":
        keep_symbols = ["r", "m_1", "I", "nu", "K", "g","omega"]
        sample, case_data = filter_case_variables(sample, case_data, keep_symbols)
        print("[Case filter] malkus: using variables {}".format(keep_symbols))

    args.active_variable_symbols = [
        var.get("symbol", "")
        for var in sample.get("variables", [])
    ]
    return sample, case_data


def filter_case_variables(sample, case_data, keep_symbols):
    variables = sample.get("variables", [])
    symbol_to_index = {var.get("symbol", ""): idx for idx, var in enumerate(variables)}
    missing = [symbol for symbol in keep_symbols if symbol not in symbol_to_index]
    if missing:
        raise ValueError("Cannot filter case variables. Missing symbols: {}".format(missing))

    keep_indices = [symbol_to_index[symbol] for symbol in keep_symbols]
    filtered_sample = copy.deepcopy(sample)
    filtered_sample["variables"] = [copy.deepcopy(variables[idx]) for idx in keep_indices]

    filtered_case_data = dict(case_data)
    filtered_case_data["X"] = np.asarray(case_data["X"])[:, keep_indices]
    if case_data.get("D_in") is not None:
        filtered_case_data["D_in"] = np.asarray(case_data["D_in"])[:, keep_indices]

    metadata = case_data.get("metadata")
    if metadata:
        filtered_metadata = copy.deepcopy(metadata)
        if "input_columns" in filtered_metadata:
            filtered_columns = [
                copy.deepcopy(filtered_metadata["input_columns"][idx])
                for idx in keep_indices
            ]
            for new_idx, col in enumerate(filtered_columns):
                col["index"] = new_idx
            filtered_metadata["input_columns"] = filtered_columns
        if "shape" in filtered_metadata:
            filtered_metadata["shape"] = copy.deepcopy(filtered_metadata["shape"])
            filtered_metadata["shape"]["X"] = list(filtered_case_data["X"].shape)
            if filtered_case_data.get("D_in") is not None:
                filtered_metadata["shape"]["D_in"] = list(filtered_case_data["D_in"].shape)
        filtered_case_data["metadata"] = filtered_metadata

    return filtered_sample, filtered_case_data


def candidate_label(index):
    return "Pi_{0}".format(index + 1)


def compute_log_pi_matrix(x_raw, candidates, eps=1e-300):
    x_raw = np.asarray(x_raw, dtype=float)
    n_rows = x_raw.shape[0]
    log_pi = np.zeros((n_rows, len(candidates)), dtype=float)
    valid = np.all(np.isfinite(x_raw), axis=1)

    for j, cand in enumerate(candidates):
        alpha = np.asarray(cand["full_alpha"], dtype=float)
        values = np.zeros(n_rows, dtype=float)
        cand_valid = np.ones(n_rows, dtype=bool)
        for col, power in enumerate(alpha):
            if abs(power) <= 1e-14:
                continue
            x_col = x_raw[:, col]
            cand_valid &= np.isfinite(x_col) & (x_col > eps)
            safe = np.where(x_col > eps, x_col, np.nan)
            values += power * np.log(safe)
        log_pi[:, j] = values
        valid &= cand_valid & np.isfinite(values)

    return log_pi, valid


def select_target(y_raw, output_index, transform):
    y_raw = np.asarray(y_raw, dtype=float)
    if y_raw.ndim == 1:
        y_col = y_raw
    else:
        if output_index < 0 or output_index >= y_raw.shape[1]:
            raise ValueError("output_index {} is outside Y shape {}".format(output_index, y_raw.shape))
        y_col = y_raw[:, output_index]

    if transform not in {"raw", "auto_log", "log"}:
        raise ValueError("Unknown target transform: {}".format(transform))

    valid = np.isfinite(y_col)
    use_log = transform == "log"

    if transform == "auto_log":
        finite_values = y_col[valid]
        all_positive = finite_values.size > 0 and np.all(finite_values > 0)
        value_span = (
            float(np.max(finite_values) / np.min(finite_values))
            if all_positive
            else None
        )
        use_log = all_positive and value_span > 1e2
        span_text = "{:.6g}".format(value_span) if value_span is not None else "not available"
        print(
            "[Target transform] auto_log: positive={0}, max/min={1}, "
            "threshold=1e2 -> use {2}".format(
                all_positive,
                span_text,
                "log" if use_log else "raw",
            )
        )

    if use_log:
        valid &= y_col > 0
        transformed = np.full(y_col.shape, np.nan, dtype=float)
        transformed[valid] = np.log(y_col[valid])
        print("[Target transform] log applied to target.")
        return transformed, valid, "log"

    print("[Target transform] target kept in raw scale; log not applied.")
    return y_col, valid, "raw"


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def random_regression_split(n_rows, train_fraction=0.8, random_seed=42):
    """Create one reproducible random split shared by Step 2 and Step 3."""
    if n_rows < 2:
        raise ValueError("At least two rows are required for an 80/20 split.")
    rng = np.random.RandomState(random_seed)
    shuffled = rng.permutation(n_rows)
    n_train = int(np.floor(float(train_fraction) * n_rows))
    n_train = min(max(n_train, 1), n_rows - 1)
    return np.sort(shuffled[:n_train]), np.sort(shuffled[n_train:])


def simple_combined_alpha(candidates, kept_indices, beta_fractions):
    a_matrix = np.array(
        [candidates[i]["full_alpha"] for i in kept_indices],
        dtype=object,
    )
    gamma = []
    for col in range(a_matrix.shape[1]):
        value = Fraction(0, 1)
        for row, beta in enumerate(beta_fractions):
            value += beta * Fraction(a_matrix[row, col]).limit_denominator(1000000)
        gamma.append(value)
    return gamma


def print_top_candidates(candidates):
    print("\n" + "=" * 80)
    print("Top-K base dimensionless numbers from model")
    print("=" * 80)
    for i, cand in enumerate(candidates):
        print(
            "{label}: score={score:.4f}, source={source}, alpha={alpha}, formula={formula}".format(
                label=candidate_label(i),
                score=cand["score"],
                source=cand.get("source", ""),
                alpha=cand["full_alpha"],
                formula=cand["formula"],
            )
        )


def deduplicate_candidates_by_alpha(candidates):
    seen = set()
    unique = []
    removed = []
    for cand in candidates:
        key = tuple(cand["full_alpha"])
        if key in seen:
            removed.append(cand)
            continue
        seen.add(key)
        unique.append(cand)
    return unique, removed


def select_independent_rank_candidates(candidates, max_rank=5):
    selected = []
    skipped = []
    for idx, cand in enumerate(candidates[:max_rank]):
        trial = selected + [idx]
        a_matrix = np.array([candidates[i]["full_alpha"] for i in trial], dtype=float)
        if np.linalg.matrix_rank(a_matrix) == len(trial):
            selected.append(idx)
        else:
            skipped.append(idx)
    return selected, skipped


def fraction_text(value):
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return "{0}/{1}".format(value.numerator, value.denominator)


def normalize_beta_direction(beta):
    beta = tuple(Fraction(x) for x in beta)
    if all(x == 0 for x in beta):
        return None
    for value in beta:
        if value != 0:
            if value < 0:
                return tuple(-x for x in beta)
            return beta
    return beta


BETA_COMPLEXITY_WEIGHT = 0.01


def beta_complexity(beta):
    score = 0.0
    for value in beta:
        value = Fraction(value)
        if value == 0:
            continue
        # ±1 is the simplest nonzero classic exponent; do not penalize it.
        # This keeps [0, 1, 0] / [0, -1, 0] strictly simpler than [0, 1/2, 0].
        if abs(value) == 1 and value.denominator == 1:
            continue
        score += abs(float(value)) + 0.05 * value.denominator
    return score


def beta_rank_score(test_r2, complexity, weight=BETA_COMPLEXITY_WEIGHT):
    """Rank Step 2/3 candidates by held-out test R2, penalized by beta complexity."""
    return float(test_r2) - float(weight) * float(complexity)


def betas_are_collinear(beta_a, beta_b):
    """True if two beta vectors are scalar multiples, including a sign flip."""
    a = [Fraction(x) for x in beta_a]
    b = [Fraction(x) for x in beta_b]
    if len(a) != len(b):
        return False
    scale = None
    for ai, bi in zip(a, b):
        if ai == 0 and bi == 0:
            continue
        if ai == 0 or bi == 0:
            return False
        ratio = ai / bi
        if scale is None:
            scale = ratio
        elif ratio != scale:
            return False
    return True


def collapse_collinear_beta_rows(rows, r2_key, max_keep):
    """Keep the best-ranked row from each collinear beta family.

    Rows are sorted by beta_sort_key first, so the first member of a
    scalar-multiple family has the best rank score (highest test R2, then
    lowest complexity). The returned rows are pairwise non-collinear.
    """
    kept = []
    for row in sorted(rows, key=lambda item: beta_sort_key(item, r2_key)):
        if any(betas_are_collinear(row["beta"], other["beta"]) for other in kept):
            continue
        kept.append(row)
        if len(kept) >= max_keep:
            break
    return kept


def design_matrix_1d(x, degree):
    x = np.asarray(x, dtype=float).reshape(-1)
    cols = [np.ones_like(x)]
    for power in range(1, degree + 1):
        cols.append(x ** power)
    return np.column_stack(cols)


def fit_poly_1d_train_test(x_train, target_train, x_test, target_test, degree):
    """Fit on the 80% training set and report train/test R2 separately."""
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    target_train = np.asarray(target_train, dtype=float).reshape(-1)
    x_test = np.asarray(x_test, dtype=float).reshape(-1)
    target_test = np.asarray(target_test, dtype=float).reshape(-1)
    if np.std(x_train) <= 1e-14:
        return None

    x_mean = float(np.mean(x_train))
    x_scale = float(np.std(x_train))
    if x_scale == 0:
        x_scale = 1.0
    train_design = design_matrix_1d((x_train - x_mean) / x_scale, degree)
    test_design = design_matrix_1d((x_test - x_mean) / x_scale, degree)
    coef, _, _, _ = np.linalg.lstsq(train_design, target_train, rcond=None)
    train_pred = np.dot(train_design, coef)
    test_pred = np.dot(test_design, coef)
    train_r2 = float(r2_score_np(target_train, train_pred))
    test_r2 = float(r2_score_np(target_test, test_pred))
    return {
        "degree": degree,
        "coef": coef,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "pred": train_pred,
        "test_pred": test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        # Keep r2 as the ranking metric for downstream compatibility.
        "r2": test_r2,
    }


def monomial_powers(n_features, degree):
    powers = []
    for exponents in itertools.product(range(degree + 1), repeat=n_features):
        total_degree = sum(exponents)
        if total_degree <= degree:
            powers.append(exponents)
    powers.sort(key=lambda item: (sum(item), item))
    return powers


def design_matrix_nd(x, degree):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    cols = []
    powers = monomial_powers(x.shape[1], degree)
    for power_tuple in powers:
        col = np.ones(x.shape[0], dtype=float)
        for feature_idx, power in enumerate(power_tuple):
            if power:
                col *= x[:, feature_idx] ** power
        cols.append(col)
    return np.column_stack(cols), powers


def fit_poly_nd_train_test(features_train, target_train, features_test, target_test, degree):
    """Fit a multivariate polynomial on train only and score both splits."""
    x_train = np.asarray(features_train, dtype=float)
    x_test = np.asarray(features_test, dtype=float)
    y_train = np.asarray(target_train, dtype=float).reshape(-1)
    y_test = np.asarray(target_test, dtype=float).reshape(-1)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1, 1)
    if (
        x_train.shape[0] != y_train.size
        or x_test.shape[0] != y_test.size
        or x_train.shape[1] != x_test.shape[1]
        or np.any(~np.isfinite(x_train))
        or np.any(~np.isfinite(x_test))
        or np.any(np.std(x_train, axis=0) <= 1e-14)
    ):
        return None

    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale == 0] = 1.0
    train_design, powers = design_matrix_nd((x_train - mean) / scale, degree)
    test_design, _ = design_matrix_nd((x_test - mean) / scale, degree)
    coef, _, _, _ = np.linalg.lstsq(train_design, y_train, rcond=None)
    train_pred = np.dot(train_design, coef)
    test_pred = np.dot(test_design, coef)
    train_r2 = float(r2_score_np(y_train, train_pred))
    test_r2 = float(r2_score_np(y_test, test_pred))
    return {
        "degree": int(degree),
        "coef": coef,
        "powers": powers,
        "mean": mean,
        "scale": scale,
        "pred": train_pred,
        "test_pred": test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "r2": test_r2,
    }


def fit_linear_nd_train_test(features_train, target_train, features_test, target_test):
    """Fit an ordinary multivariate linear model on raw features."""
    x_train = np.asarray(features_train, dtype=float)
    x_test = np.asarray(features_test, dtype=float)
    y_train = np.asarray(target_train, dtype=float).reshape(-1)
    y_test = np.asarray(target_test, dtype=float).reshape(-1)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1, 1)
    if (
        x_train.shape[0] != y_train.size
        or x_test.shape[0] != y_test.size
        or x_train.shape[1] != x_test.shape[1]
        or np.any(~np.isfinite(x_train))
        or np.any(~np.isfinite(x_test))
    ):
        return None

    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0)
    if np.any(feature_scale <= 1e-14):
        return None
    train_standardized = (x_train - feature_mean) / feature_scale
    test_standardized = (x_test - feature_mean) / feature_scale
    train_design = np.column_stack([np.ones(x_train.shape[0]), train_standardized])
    test_design = np.column_stack([np.ones(x_test.shape[0]), test_standardized])
    coef, _, _, _ = np.linalg.lstsq(train_design, y_train, rcond=None)
    train_pred = np.dot(train_design, coef)
    test_pred = np.dot(test_design, coef)
    train_r2 = float(r2_score_np(y_train, train_pred))
    test_r2 = float(r2_score_np(y_test, test_pred))
    return {
        "degree": 1,
        "coef": coef,
        "mean": feature_mean,
        "scale": feature_scale,
        "pred": train_pred,
        "test_pred": test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "r2": test_r2,
    }


def total_log_pi(log_pi_selected, beta):
    beta_float = np.array([float(x) for x in beta], dtype=float)
    return np.dot(log_pi_selected, beta_float)


def beta_sort_key(row, r2_key="linear_test_r2"):
    test_r2 = float(row[r2_key])
    complexity = float(row["complexity"])
    return (-beta_rank_score(test_r2, complexity), -test_r2, complexity)


def covariance_screening_stats(log_pi_selected, target):
    x = np.asarray(log_pi_selected, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    x_mean = np.mean(x, axis=0)
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    n_rows = max(1, x.shape[0])
    sigma = np.dot(x_centered.T, x_centered) / n_rows
    c = np.dot(x_centered.T, y_centered) / n_rows
    y_var = float(np.dot(y_centered, y_centered) / n_rows)
    return x_mean, y_mean, sigma, c, y_var


def make_linear_screening_row(
    beta,
    train_r2,
    test_r2,
    complexity,
    mean_x=None,
    y_mean=None,
    cov_xy=None,
    var_x=None,
):
    linear = {
        "degree": 1,
        "r2": float(test_r2),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
    }
    if mean_x is not None and y_mean is not None and cov_xy is not None and var_x is not None and var_x > 1e-30:
        slope = float(cov_xy / var_x)
        linear["slope"] = slope
        linear["intercept"] = float(y_mean - slope * mean_x)
    return {
        "beta": tuple(Fraction(x) for x in beta),
        "linear": linear,
        "linear_r2": float(test_r2),
        "linear_train_r2": float(train_r2),
        "linear_test_r2": float(test_r2),
        "complexity": float(complexity),
    }


def push_screening_row(heap, rows, retain_top_n, counter, row):
    if retain_top_n is None or retain_top_n <= 0:
        rows.append(row)
        return counter + 1

    heap_key = (
        beta_rank_score(row["linear_test_r2"], row["complexity"]),
        row["linear_test_r2"],
        -row["complexity"],
        counter,
    )
    if len(heap) < retain_top_n:
        heapq.heappush(heap, (heap_key, row))
    elif heap_key > heap[0][0]:
        heapq.heapreplace(heap, (heap_key, row))
    return counter + 1


def rows_from_screening_storage(heap, rows, retain_top_n):
    if retain_top_n is not None and retain_top_n > 0:
        rows = [item[1] for item in heap]
    rows.sort(key=beta_sort_key)
    return rows


def evaluate_beta_screening_batch(
    beta_items,
    x_mean,
    y_mean,
    sigma,
    c,
    y_var,
    log_pi_test,
    target_test,
):
    if not beta_items or y_var <= 1e-30:
        return []

    log_pi_test = np.asarray(log_pi_test, dtype=float)
    target_test = np.asarray(target_test, dtype=float).reshape(-1)
    beta_float = np.array([[float(value) for value in beta] for beta in beta_items], dtype=float)
    mean_x = np.dot(beta_float, x_mean)
    cov_xy = np.dot(beta_float, c)
    var_x = np.einsum("qi,ij,qj->q", beta_float, sigma, beta_float)
    valid = np.isfinite(var_x) & np.isfinite(cov_xy) & (var_x > 1e-30)
    if not np.any(valid):
        return []

    train_r2 = np.zeros(len(beta_items), dtype=float)
    train_r2[valid] = (cov_xy[valid] ** 2) / (var_x[valid] * y_var)
    train_r2 = np.clip(train_r2, 0.0, 1.0)

    slope = np.zeros(len(beta_items), dtype=float)
    intercept = np.zeros(len(beta_items), dtype=float)
    slope[valid] = cov_xy[valid] / var_x[valid]
    intercept[valid] = y_mean - slope[valid] * mean_x[valid]
    test_total = np.dot(log_pi_test, beta_float.T)
    test_pred = intercept.reshape(1, -1) + test_total * slope.reshape(1, -1)
    test_ss_res = np.sum(
        (np.asarray(target_test, dtype=float).reshape(-1, 1) - test_pred) ** 2,
        axis=0,
    )
    test_ss_tot = float(
        np.sum((target_test - np.mean(target_test)) ** 2)
    )
    if test_ss_tot <= 0.0:
        test_r2 = np.zeros(len(beta_items), dtype=float)
    else:
        test_r2 = 1.0 - test_ss_res / test_ss_tot

    rows = []
    for local_idx in np.where(valid)[0]:
        beta = beta_items[local_idx]
        row = make_linear_screening_row(
            beta,
            train_r2[local_idx],
            test_r2[local_idx],
            beta_complexity(beta),
            mean_x=mean_x[local_idx],
            y_mean=y_mean,
            cov_xy=cov_xy[local_idx],
            var_x=var_x[local_idx],
        )
        rows.append(row)
    return rows


def unique_fraction_values(values):
    out = []
    seen = set()
    for value in values:
        frac = Fraction(value)
        if frac in seen:
            continue
        seen.add(frac)
        out.append(frac)
    return out


def search_classic_exponent_products_slow(
    log_pi_train,
    target_train,
    log_pi_test,
    target_test,
    exponents,
    show_progress=False,
):
    seen = set()
    rows = []
    n_base = log_pi_train.shape[1]
    total = len(exponents) ** n_base
    beta_iter = itertools.product(exponents, repeat=n_base)
    for beta_raw in progress_iter(
        beta_iter,
        total=total,
        desc="Step 2 classic exponent search",
        enabled=show_progress,
    ):
        beta = normalize_beta_direction(beta_raw)
        if beta is None:
            continue
        if beta in seen:
            continue
        seen.add(beta)

        x_train = total_log_pi(log_pi_train, beta)
        x_test = total_log_pi(log_pi_test, beta)
        fit = fit_poly_1d_train_test(
            x_train,
            target_train,
            x_test,
            target_test,
            degree=1,
        )
        if fit is None:
            continue
        rows.append(
            {
                "beta": beta,
                "log_total": x_train,
                "log_total_test": x_test,
                "linear": fit,
                "linear_r2": fit["test_r2"],
                "linear_train_r2": fit["train_r2"],
                "linear_test_r2": fit["test_r2"],
                "complexity": beta_complexity(beta),
            }
        )

    rows.sort(key=lambda row: beta_sort_key(row, "linear_test_r2"))
    return rows


def search_classic_exponent_products_fast(
    log_pi_train,
    target_train,
    log_pi_test,
    target_test,
    exponents,
    max_active=None,
    retain_top_n=None,
    chunk_size=50000,
    branch_bound_margin=1e-12,
    show_progress=False,
):
    x_mean, y_mean, sigma, c, y_var = covariance_screening_stats(log_pi_train, target_train)
    n_base = log_pi_train.shape[1]
    exponents = unique_fraction_values(exponents)
    nonzero_exponents = [value for value in exponents if value != 0]
    chunk_size = max(1, int(chunk_size))
    max_active = None if max_active is None or int(max_active) <= 0 else min(int(max_active), n_base)

    heap = []
    rows = []
    counter = 0
    best_r2 = 0.0
    support_checked = 0
    support_skipped = 0
    beta_checked = 0
    beta_kept = 0

    def consume_batch(batch, active_indices=None):
        nonlocal counter, best_r2, beta_kept
        if not batch:
            return
        if active_indices is None:
            batch_rows = evaluate_beta_screening_batch(
                batch,
                x_mean,
                y_mean,
                sigma,
                c,
                y_var,
                log_pi_test,
                target_test,
            )
        else:
            idx = list(active_indices)
            batch_rows = evaluate_beta_screening_batch(
                batch,
                x_mean[idx],
                y_mean,
                sigma[np.ix_(idx, idx)],
                c[idx],
                y_var,
                log_pi_test[:, idx],
                target_test,
            )
        for row in batch_rows:
            beta_kept += 1
            if row["linear_test_r2"] > best_r2:
                best_r2 = row["linear_test_r2"]
            if active_indices is not None:
                full_beta = [Fraction(0, 1)] * n_base
                for local_i, global_i in enumerate(active_indices):
                    full_beta[global_i] = row["beta"][local_i]
                row["beta"] = tuple(full_beta)
                row["complexity"] = beta_complexity(row["beta"])
            counter = push_screening_row(heap, rows, retain_top_n, counter, row)

    if max_active is None or max_active >= n_base:
        total = len(exponents) ** n_base
        beta_iter = itertools.product(exponents, repeat=n_base)
        batch = []
        seen = set()
        for beta_raw in progress_iter(
            beta_iter,
            total=total,
            desc="Step 2 accelerated classic exponent search",
            enabled=show_progress,
        ):
            beta_checked += 1
            beta = normalize_beta_direction(beta_raw)
            if beta is None or beta in seen:
                continue
            seen.add(beta)
            batch.append(beta)
            if len(batch) >= chunk_size:
                consume_batch(batch)
                batch = []
        consume_batch(batch)
    else:
        support_sizes = range(1, max_active + 1)
        supports = [
            support
            for size in support_sizes
            for support in itertools.combinations(range(n_base), size)
        ]
        for support in progress_iter(
            supports,
            total=len(supports),
            desc="Step 2 accelerated support search",
            enabled=show_progress,
        ):
            support_checked += 1
            batch = []
            seen = set()
            for beta_raw in itertools.product(nonzero_exponents, repeat=len(support)):
                beta_checked += 1
                beta = normalize_beta_direction(beta_raw)
                if beta is None or beta in seen:
                    continue
                seen.add(beta)
                batch.append(beta)
                if len(batch) >= chunk_size:
                    consume_batch(batch, active_indices=support)
                    batch = []
            consume_batch(batch, active_indices=support)

    rows = rows_from_screening_storage(heap, rows, retain_top_n)
    print(
        "[Accelerated Step 2] beta checked={0}, valid={1}, retained={2}, supports checked={3}, supports pruned={4}".format(
            beta_checked,
            beta_kept,
            len(rows),
            support_checked,
            support_skipped,
        )
    )
    return rows


def search_classic_exponent_products(
    log_pi_train,
    target_train,
    log_pi_test,
    target_test,
    exponents,
    show_progress=False,
    accelerated=False,
    max_active=None,
    retain_top_n=None,
    chunk_size=50000,
    branch_bound_margin=1e-12,
):
    if not accelerated:
        return search_classic_exponent_products_slow(
            log_pi_train,
            target_train,
            log_pi_test,
            target_test,
            exponents,
            show_progress=show_progress,
        )
    return search_classic_exponent_products_fast(
        log_pi_train,
        target_train,
        log_pi_test,
        target_test,
        exponents,
        max_active=max_active,
        retain_top_n=retain_top_n,
        chunk_size=chunk_size,
        branch_bound_margin=branch_bound_margin,
        show_progress=show_progress,
    )


def refine_with_polynomial(
    rows,
    target_train,
    target_test,
    max_degree,
    top_m=None,
    show_progress=False,
    log_pi_train=None,
    log_pi_test=None,
):
    refined = []
    selected_rows = rows if top_m is None or top_m <= 0 else rows[:top_m]
    for row in progress_iter(
        selected_rows,
        total=len(selected_rows),
        desc="Step 3 polynomial refinement",
        enabled=show_progress,
    ):
        if "log_total" in row:
            x_train = row["log_total"]
            x_test = row["log_total_test"]
        else:
            if log_pi_train is None or log_pi_test is None:
                raise ValueError("log_pi_train and log_pi_test are required when rows do not store log_total.")
            x_train = total_log_pi(log_pi_train, row["beta"])
            x_test = total_log_pi(log_pi_test, row["beta"])

        best = row["linear"]
        best_degree = 1
        for degree in range(2, max_degree + 1):
            fit = fit_poly_1d_train_test(
                x_train,
                target_train,
                x_test,
                target_test,
                degree=degree,
            )
            if fit is not None and fit["test_r2"] > best["test_r2"]:
                best = fit
                best_degree = degree
        item = dict(row)
        item["poly"] = best
        item["poly_degree"] = best_degree
        item["poly_r2"] = best["test_r2"]
        item["poly_train_r2"] = best["train_r2"]
        item["poly_test_r2"] = best["test_r2"]
        refined.append(item)
    refined.sort(key=lambda row: beta_sort_key(row, "poly_test_r2"))
    return refined


def scale_beta(beta, scale):
    scale = Fraction(scale)
    return tuple(Fraction(x) * scale for x in beta)


def collinear_scale_factors(exponents):
    """Positive and negative nonzero scales from classic exponents."""
    positives = []
    seen = set()
    for value in exponents:
        frac = abs(Fraction(value))
        if frac == 0 or frac in seen:
            continue
        seen.add(frac)
        positives.append(frac)
    return [-frac for frac in positives] + positives


def pi_from_log_pi(log_pi_selected, beta):
    log_total = total_log_pi(log_pi_selected, beta)
    return np.exp(np.clip(np.asarray(log_total, dtype=float), -700.0, 700.0))


def refine_with_raw_pi_polynomial(
    direction_rows,
    log_pi_train,
    target_train,
    log_pi_test,
    target_test,
    scales,
    max_degree,
    show_progress=False,
):
    """Polynomial regression on Pi itself for collinear multiples of one beta."""
    jobs = []
    seen = set()
    for row in direction_rows:
        base = tuple(Fraction(x) for x in row["beta"])
        for scale in scales:
            beta = scale_beta(base, scale)
            if all(x == 0 for x in beta) or beta in seen:
                continue
            seen.add(beta)
            jobs.append((row, beta, scale))

    refined = []
    for row, beta, scale in progress_iter(
        jobs,
        total=len(jobs),
        desc="Step 4 raw-pi polynomial",
        enabled=show_progress,
    ):
        pi_train = pi_from_log_pi(log_pi_train, beta)
        pi_test = pi_from_log_pi(log_pi_test, beta)
        if (
            np.any(~np.isfinite(pi_train))
            or np.any(~np.isfinite(pi_test))
            or np.std(pi_train) <= 1e-14
        ):
            continue
        best = None
        best_degree = 1
        for degree in range(1, max_degree + 1):
            fit = fit_poly_1d_train_test(
                pi_train,
                target_train,
                pi_test,
                target_test,
                degree=degree,
            )
            if fit is None:
                continue
            if best is None or fit["test_r2"] > best["test_r2"]:
                best = fit
                best_degree = degree
        if best is None:
            continue
        item = dict(row)
        item.pop("log_total", None)
        item.pop("log_total_test", None)
        item["beta"] = beta
        item["beta_scale"] = scale
        item["complexity"] = beta_complexity(beta)
        item["raw_poly"] = best
        item["raw_poly_degree"] = best_degree
        item["raw_poly_r2"] = best["test_r2"]
        item["raw_poly_train_r2"] = best["train_r2"]
        item["raw_poly_test_r2"] = best["test_r2"]
        finite_train = pi_train[np.isfinite(pi_train)]
        item["pi_train_min"] = float(np.min(finite_train))
        item["pi_train_max"] = float(np.max(finite_train))
        refined.append(item)
    refined.sort(key=lambda row: beta_sort_key(row, "raw_poly_test_r2"))
    return refined


def alpha_vector_complexity(alpha):
    arr = np.array(alpha, dtype=float)
    return float(np.sum(np.abs(arr)) + 0.1 * np.count_nonzero(arr))


def term_alpha(candidates, term):
    total = np.zeros(len(candidates[term[0]]["full_alpha"]), dtype=float)
    for idx in term:
        total += np.array(candidates[idx]["full_alpha"], dtype=float)
    return total


def term_label(term):
    counts = {}
    for idx in term:
        counts[idx] = counts.get(idx, 0) + 1
    parts = []
    for idx in sorted(counts):
        label = candidate_label(idx)
        if counts[idx] == 1:
            parts.append(label)
        else:
            parts.append("{0}^{1}".format(label, counts[idx]))
    return "*".join(parts)


def term_formula(candidates, term):
    counts = {}
    for idx in term:
        counts[idx] = counts.get(idx, 0) + 1
    parts = []
    for idx in sorted(counts):
        formula = candidates[idx]["formula"]
        if counts[idx] == 1:
            parts.append("({0})".format(formula))
        else:
            parts.append("({0})^{1}".format(formula, counts[idx]))
    return " * ".join(parts) if parts else "1"


def step6_log_pi_values(log_pi, feature_spec):
    """Evaluate one two-group PI feature in log coordinates."""
    kind = feature_spec["kind"]
    if kind == "base":
        return np.asarray(log_pi[:, int(feature_spec["candidate_index"])], dtype=float)
    if kind == "step3":
        selected_indices = [int(index) for index in feature_spec["selected_indices"]]
        beta = [Fraction(value) for value in feature_spec["beta"]]
        log_values = total_log_pi(log_pi[:, selected_indices], beta)
        return float(feature_spec.get("power", 1)) * log_values
    raise ValueError("Unknown Step-6 PI feature kind: {0}".format(kind))


def step6_raw_pi_values(log_pi, feature_spec):
    """Evaluate one two-group PI feature without applying a logarithm."""
    return np.exp(np.clip(step6_log_pi_values(log_pi, feature_spec), -700.0, 700.0))


def score_selected_two_pi_log_polynomial(
    row,
    log_pi_train,
    target_train,
    log_pi_test,
    target_test,
    max_degree,
):
    """Keep the selected Pi pair, but score it with poly(log Pi)."""
    log_train = np.column_stack(
        [step6_log_pi_values(log_pi_train, spec) for spec in row["feature_specs"]]
    )
    log_test = np.column_stack(
        [step6_log_pi_values(log_pi_test, spec) for spec in row["feature_specs"]]
    )
    best = None
    max_degree = max(1, int(max_degree))
    for degree in range(1, max_degree + 1):
        fit = fit_poly_nd_train_test(
            log_train,
            target_train,
            log_test,
            target_test,
            degree=degree,
        )
        if fit is None:
            continue
        if best is None or fit["test_r2"] > best["test_r2"]:
            best = fit
    if best is None:
        return row
    scored = dict(row)
    scored["poly"] = best
    scored["poly_r2"] = best["test_r2"]
    scored["poly_train_r2"] = best["train_r2"]
    scored["poly_test_r2"] = best["test_r2"]
    scored["poly_degree"] = int(best["degree"])
    scored["regression_input"] = "log(pi)"
    return scored


def search_two_pi_models(
    candidates,
    selected,
    step3_row,
    log_pi_train,
    target_train,
    log_pi_test,
    target_test,
    args,
):
    """Select a two-pi pair by raw linear R2, then score the winner with poly(log Pi)."""
    print("\n" + "=" * 80)
    print("Step 5 | Raw two-pi linear regression over four prescribed PI features")
    print("=" * 80)
    if len(selected) < 2:
        print("Step 5 skipped: fewer than two selected base PI groups are available.")
        return [], False

    base_indices = [int(selected[0]), int(selected[1])]
    beta = tuple(Fraction(value) for value in step3_row["beta"])
    combined_alpha = combined_alpha_for_search(candidates, selected, beta)
    combined_formula = total_formula_from_beta(candidates, selected, beta)
    inverse_alpha = [-float(value) for value in combined_alpha]

    feature_defs = [
        {
            "label": candidate_label(base_indices[0]),
            "formula": candidates[base_indices[0]]["formula"],
            "alpha": list(candidates[base_indices[0]]["full_alpha"]),
            "spec": {"kind": "base", "candidate_index": base_indices[0]},
        },
        {
            "label": candidate_label(base_indices[1]),
            "formula": candidates[base_indices[1]]["formula"],
            "alpha": list(candidates[base_indices[1]]["full_alpha"]),
            "spec": {"kind": "base", "candidate_index": base_indices[1]},
        },
        {
            "label": "Pi_step3",
            "formula": combined_formula,
            "alpha": [float(value) for value in combined_alpha],
            "spec": {
                "kind": "step3",
                "selected_indices": tuple(int(index) for index in selected),
                "beta": beta,
                "power": 1,
            },
        },
        {
            "label": "1/Pi_step3",
            "formula": "1 / ({0})".format(combined_formula),
            "alpha": inverse_alpha,
            "spec": {
                "kind": "step3",
                "selected_indices": tuple(int(index) for index in selected),
                "beta": beta,
                "power": -1,
            },
        },
    ]

    print("candidate PI pool = {0}".format([item["label"] for item in feature_defs]))
    print("pair selection    = raw PI linear regression")
    print("final scoring     = poly(log Pi) on the selected pair")
    train_features = [step6_raw_pi_values(log_pi_train, item["spec"]) for item in feature_defs]
    test_features = [step6_raw_pi_values(log_pi_test, item["spec"]) for item in feature_defs]

    rows = []
    for pair in itertools.combinations(range(4), 2):
        pair_train = np.column_stack([train_features[index] for index in pair])
        pair_test = np.column_stack([test_features[index] for index in pair])
        linear = fit_linear_nd_train_test(
            pair_train,
            target_train,
            pair_test,
            target_test,
        )
        if linear is None:
            print(
                "skip pair ({0}, {1}) | constant or invalid raw PI feature".format(
                    feature_defs[pair[0]]["label"],
                    feature_defs[pair[1]]["label"],
                )
            )
            continue
        chosen = [feature_defs[index] for index in pair]
        complexity = sum(alpha_vector_complexity(item["alpha"]) for item in chosen)
        rows.append(
            {
                "pair": tuple(pair),
                "pi_labels": tuple(item["label"] for item in chosen),
                "pi_formulas": tuple(item["formula"] for item in chosen),
                "pi_alphas": tuple(tuple(item["alpha"]) for item in chosen),
                "feature_specs": tuple(item["spec"] for item in chosen),
                "mode": "selected_step3_raw_linear",
                "linear": linear,
                "linear_r2": linear["test_r2"],
                "linear_train_r2": linear["train_r2"],
                "linear_test_r2": linear["test_r2"],
                # Placeholder poly fields; the selected pair is rescored with poly(log Pi).
                "poly": linear,
                "poly_r2": linear["test_r2"],
                "poly_train_r2": linear["train_r2"],
                "poly_test_r2": linear["test_r2"],
                "poly_degree": 1,
                "base_score": 0.0,
                "complexity": complexity,
                "composite_score": linear["test_r2"],
            }
        )

    rows.sort(key=lambda row: -row["linear_test_r2"])
    for rank, row in enumerate(rows, start=1):
        print(
            "Rank {0} | pair ({1}, {2}) | train R2={3:.6f} | test R2={4:.6f}".format(
                rank,
                row["pi_labels"][0],
                row["pi_labels"][1],
                row["linear_train_r2"],
                row["linear_test_r2"],
            )
        )
    if rows:
        rows[0] = score_selected_two_pi_log_polynomial(
            rows[0],
            log_pi_train,
            target_train,
            log_pi_test,
            target_test,
            max_degree=getattr(args, "polynomial_max_degree", 5),
        )
        print(
            "Selected pair poly(log Pi): degree={0} | train R2={1:.6f} | test R2={2:.6f}".format(
                rows[0].get("poly_degree", 1),
                rows[0].get("poly_train_r2", rows[0]["linear_train_r2"]),
                rows[0].get("poly_test_r2", rows[0]["linear_test_r2"]),
            )
        )
    return rows, False


def combined_alpha_for_search(candidates, selected_indices, beta):
    beta_fractions = [Fraction(x) for x in beta]
    return simple_combined_alpha(candidates, selected_indices, beta_fractions)


def total_formula_from_beta(candidates, selected_indices, beta):
    parts = []
    for idx, exponent in zip(selected_indices, beta):
        exponent = Fraction(exponent)
        if exponent == 0:
            continue
        formula = candidates[idx]["formula"]
        if exponent == 1:
            parts.append("({0})".format(formula))
        else:
            parts.append("({0})^{1}".format(formula, fraction_text(exponent)))
    return " * ".join(parts) if parts else "1"


def print_independent_candidates(candidates, selected, skipped, rank_top_n):
    print("\n" + "=" * 80)
    print("Step 1 | Independent base pi from Rank 1-{0}".format(rank_top_n))
    print("=" * 80)
    for idx in selected:
        print(
            "selected {label}: score={score:.4f}, alpha={alpha}, formula={formula}".format(
                label=candidate_label(idx),
                score=candidates[idx]["score"],
                alpha=candidates[idx]["full_alpha"],
                formula=candidates[idx]["formula"],
            )
        )
    for idx in skipped:
        print(
            "skipped  {label}: rank dependent, alpha={alpha}, formula={formula}".format(
                label=candidate_label(idx),
                alpha=candidates[idx]["full_alpha"],
                formula=candidates[idx]["formula"],
            )
        )
    if selected:
        a_matrix = np.array([candidates[i]["full_alpha"] for i in selected], dtype=float)
        print("rank(A) = {} / {}".format(np.linalg.matrix_rank(a_matrix), len(selected)))


def print_search_results(title, rows, candidates, selected, sample, metric_key, max_print=5, header_notes=None):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if header_notes:
        for line in header_notes:
            print(line)
    if metric_key in {"linear_r2", "poly_r2"}:
        print(
            "ranking score = test R2 - {0:.2f} * complexity".format(BETA_COMPLEXITY_WEIGHT)
        )
        print("display: collinear beta (scalar multiples) collapsed to the best rank score")
        print("listed beta are pairwise non-collinear")
        r2_key = "linear_test_r2" if metric_key == "linear_r2" else "poly_test_r2"
        rows = collapse_collinear_beta_rows(rows, r2_key, max_print)
    elif metric_key == "raw_poly_r2" and not header_notes:
        print("regression x     = Pi (not log Pi)")
        print(
            "ranking score = test R2 - {0:.2f} * complexity".format(BETA_COMPLEXITY_WEIGHT)
        )
    if not rows:
        print("No valid result.")
        return
    for rank, row in enumerate(rows[:max_print], start=1):
        gamma = combined_alpha_for_search(candidates, selected, row["beta"])
        print("\nRank {0}".format(rank))
        if metric_key == "linear_r2":
            test_r2 = row["linear_test_r2"]
            print("  train R2       = {:.6f}".format(row["linear_train_r2"]))
            print("  test R2        = {:.6f}".format(test_r2))
            print("  complexity     = {:.6f}".format(row["complexity"]))
            print(
                "  rank score     = {0:.6f}  (test R2 - {1:.2f} * complexity)".format(
                    beta_rank_score(test_r2, row["complexity"]),
                    BETA_COMPLEXITY_WEIGHT,
                )
            )
        elif metric_key == "poly_r2":
            test_r2 = row["poly_test_r2"]
            print("  train R2       = {:.6f}".format(row["poly_train_r2"]))
            print("  test R2        = {:.6f}".format(test_r2))
            print("  complexity     = {:.6f}".format(row["complexity"]))
            print(
                "  rank score     = {0:.6f}  (test R2 - {1:.2f} * complexity)".format(
                    beta_rank_score(test_r2, row["complexity"]),
                    BETA_COMPLEXITY_WEIGHT,
                )
            )
        elif metric_key == "raw_poly_r2":
            test_r2 = row["raw_poly_test_r2"]
            print("  train R2       = {:.6f}".format(row["raw_poly_train_r2"]))
            print("  test R2        = {:.6f}".format(test_r2))
            print("  complexity     = {:.6f}".format(row["complexity"]))
            print(
                "  rank score     = {0:.6f}  (test R2 - {1:.2f} * complexity)".format(
                    beta_rank_score(test_r2, row["complexity"]),
                    BETA_COMPLEXITY_WEIGHT,
                )
            )
            print("  beta scale     = {}".format(fraction_text(row.get("beta_scale", 1))))
            if "pi_train_min" in row and "pi_train_max" in row:
                print(
                    "  Pi train range = [{0:.6g}, {1:.6g}]".format(
                        row["pi_train_min"],
                        row["pi_train_max"],
                    )
                )
        else:
            print("  R2             = {:.6f}".format(row[metric_key]))
        print("  beta           = {}".format([fraction_text(x) for x in row["beta"]]))
        print("  beta formula   = {}".format(total_formula_from_beta(candidates, selected, row["beta"])))
        print("  combined alpha = {}".format([fraction_text(x) for x in gamma]))
        print("  combined form  = {}".format(alpha_to_formula(sample, gamma)))
        if metric_key == "poly_r2":
            print("  poly degree    = {}".format(row["poly_degree"]))
        if metric_key == "raw_poly_r2":
            print("  poly degree    = {}".format(row["raw_poly_degree"]))


def print_two_pi_result(row, candidates):
    print("\n" + "=" * 80)
    print("Best two-pi model")
    print("=" * 80)
    if row is None:
        print("No valid two-pi model.")
        return
    if row.get("mode") == "selected_step3_raw_linear":
        print("pi pair         = ({0}, {1})".format(*row["pi_labels"]))
        print("selection       = raw multivariate linear regression")
        print("selection train R2 = {:.6f}".format(row["linear_train_r2"]))
        print("selection test R2  = {:.6f}".format(row["linear_test_r2"]))
        print("method          = poly(log Pi)")
        print("regression x    = log(Pi)")
        print("train R2        = {:.6f}".format(row.get("poly_train_r2", row["linear_train_r2"])))
        print("test R2         = {:.6f}".format(row.get("poly_test_r2", row["linear_test_r2"])))
        print("poly degree     = {}".format(row.get("poly_degree", 1)))
        for label, alpha, formula in zip(
            row["pi_labels"], row["pi_alphas"], row["pi_formulas"]
        ):
            print("{0} alpha       = {1}".format(label, [float(x) for x in alpha]))
            print("{0} formula     = {1}".format(label, formula))
        return
    if "term_pair" in row:
        left, right = row["term_pair"]
        print("pi pair         = ({0}, {1})".format(term_label(left), term_label(right)))
        print("mode            = {}".format(row.get("mode", "product")))
    else:
        i, j = row["pair"]
        print("pi pair         = ({0}, {1})".format(candidate_label(i), candidate_label(j)))
    print("composite score = {:.6f}".format(row["composite_score"]))
    print("base score      = {:.6f}".format(row["base_score"]))
    print("complexity      = {:.6f}".format(row["complexity"]))
    print("Step 2 R2       = {:.6f}".format(row["linear_r2"]))
    print("Step 3 R2       = {:.6f}".format(row["poly_r2"]))
    print("poly degree     = {}".format(row["poly_degree"]))
    if "term_pair" in row:
        for term in row["term_pair"]:
            print("{0} alpha       = {1}".format(term_label(term), [float(x) for x in term_alpha(candidates, term)]))
            print("{0} formula     = {1}".format(term_label(term), term_formula(candidates, term)))
    else:
        print("{0} alpha       = {1}".format(candidate_label(i), candidates[i]["full_alpha"]))
        print("{0} formula     = {1}".format(candidate_label(i), candidates[i]["formula"]))
        print("{0} alpha       = {1}".format(candidate_label(j), candidates[j]["full_alpha"]))
        print("{0} formula     = {1}".format(candidate_label(j), candidates[j]["formula"]))


class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def output_log_path(args):
    safe_case_name = safe_path_component(str(args.case).strip() or "unknown_case")
    out_dir = parameter_output_dir(safe_case_name)
    return out_dir / "parameter_optimize_output{0}.txt".format(target_file_suffix(args))


def parameter_output_dir(case_name):
    safe_case_name = safe_path_component(case_name)
    out_dir = Path("results_save") / safe_case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def target_file_suffix(args):
    if canonical_case_name(args.case) != "wall_model":
        return ""
    label = getattr(args, "output_label", None) or "output_{0}".format(args.output_index)
    return "_target_{0}_{1}".format(int(args.output_index), safe_path_component(label))


def raw_candidates_cache_path(args):
    return parameter_output_dir(args.case) / "raw_candidates.pkl"


def parameter_result_cache_path(args):
    return parameter_output_dir(args.case) / "parameter_optimize_result{0}.pkl".format(
        target_file_suffix(args)
    )


def raw_candidates_cache_metadata(args, source_top_k):
    return {
        "case": str(args.case),
        "source_top_k": int(source_top_k),
        "requested_top_n": int(max(source_top_k * 3, source_top_k)),
        "subset_size_penalty": float(args.subset_size_penalty),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_name": str(args.checkpoint_name),
        "active_variable_symbols": list(getattr(args, "active_variable_symbols", [])),
    }


def parameter_result_cache_metadata(args, source_top_k):
    classic_exponents = parse_fraction_list(
        args.classic_exponents,
        [-3, -2, -1, Fraction(-1, 2), 0, Fraction(1, 2), 1, 2, 3],
    )
    return {
        "case": str(args.case),
        "source_top_k": int(source_top_k),
        "top_k": int(args.top_k),
        "independent_rank_top_n": int(args.independent_rank_top_n),
        "rank_source_top_k": None if args.rank_source_top_k is None else int(args.rank_source_top_k),
        "classic_exponents": [str(x) for x in classic_exponents],
        "r2_threshold": float(args.r2_threshold),
        "polynomial_max_degree": int(args.polynomial_max_degree),
        "search_refine_top_m": int(args.search_refine_top_m),
        "two_pi_exhaustive_search": bool(args.two_pi_exhaustive_search),
        "two_pi_complexity_weight": float(args.two_pi_complexity_weight),
        "subset_size_penalty": float(args.subset_size_penalty),
        "output_index": int(args.output_index),
        "output_label": str(getattr(args, "output_label", "")),
        "target_transform": str(args.target_transform),
        "accelerated_min_rank": int(args.accelerated_min_rank),
        "accelerated_refine_top_m": int(args.accelerated_refine_top_m),
        "accelerated_search_chunk_size": int(args.accelerated_search_chunk_size),
        "classic_max_active": int(args.classic_max_active),
        "branch_bound_margin": float(args.branch_bound_margin),
        "regression_train_fraction": 0.8,
        "regression_split_seed": 42,
        "step2_step3_rank_metric": "test_r2",
        "step6_search_version": "four_pi_raw_linear_log_poly_score_v1",
        "active_variable_symbols": list(getattr(args, "active_variable_symbols", [])),
    }


def load_raw_candidates_cache(args, source_top_k):
    path = raw_candidates_cache_path(args)
    if args.force_retain or not path.exists():
        return None

    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print("[Cache] Failed to read raw_candidates cache {0}: {1}".format(path, exc))
        return None

    expected = raw_candidates_cache_metadata(args, source_top_k)
    saved = payload.get("metadata", {})
    if saved != expected:
        print("[Cache] raw_candidates metadata mismatch. Recomputing.")
        return None

    raw_candidates = payload.get("raw_candidates")
    if not raw_candidates:
        print("[Cache] raw_candidates cache is empty. Recomputing.")
        return None

    print("[Cache] Loaded raw_candidates from {0}".format(path))
    return raw_candidates


def save_raw_candidates_cache(args, source_top_k, raw_candidates):
    path = raw_candidates_cache_path(args)
    payload = {
        "metadata": raw_candidates_cache_metadata(args, source_top_k),
        "raw_candidates": raw_candidates,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print("[Cache] Saved raw_candidates to {0}".format(path))


def minimal_search_row(row, metric_key):
    out = {
        "beta": tuple(Fraction(x) for x in row["beta"]),
        "linear_r2": float(row.get("linear_r2", 0.0)),
        "linear_train_r2": float(row.get("linear_train_r2", row.get("linear_r2", 0.0))),
        "linear_test_r2": float(row.get("linear_test_r2", row.get("linear_r2", 0.0))),
        "complexity": float(row.get("complexity", 0.0)),
        "metric_key": metric_key,
        "metric_value": float(row.get(metric_key, row.get("linear_r2", 0.0))),
    }
    if "poly_r2" in row:
        out["poly_r2"] = float(row["poly_r2"])
        out["poly_train_r2"] = float(row.get("poly_train_r2", row["poly_r2"]))
        out["poly_test_r2"] = float(row.get("poly_test_r2", row["poly_r2"]))
        out["poly_degree"] = int(row.get("poly_degree", 1))
    if "raw_poly_r2" in row:
        out["raw_poly_r2"] = float(row["raw_poly_r2"])
        out["raw_poly_train_r2"] = float(row.get("raw_poly_train_r2", row["raw_poly_r2"]))
        out["raw_poly_test_r2"] = float(row.get("raw_poly_test_r2", row["raw_poly_r2"]))
        out["raw_poly_degree"] = int(row.get("raw_poly_degree", 1))
        if "beta_scale" in row:
            out["beta_scale"] = Fraction(row["beta_scale"])
    return out


def minimal_two_pi_row(row):
    if "term_pair" in row:
        pair = tuple(tuple(int(x) for x in term) for term in row["term_pair"])
    else:
        pair = tuple(int(x) for x in row["pair"])
    out = {
        "pair": pair,
        "linear_r2": float(row.get("linear_r2", 0.0)),
        "linear_train_r2": float(row.get("linear_train_r2", row.get("linear_r2", 0.0))),
        "linear_test_r2": float(row.get("linear_test_r2", row.get("linear_r2", 0.0))),
        "poly_r2": float(row.get("poly_r2", 0.0)),
        "poly_train_r2": float(row.get("poly_train_r2", row.get("poly_r2", 0.0))),
        "poly_test_r2": float(row.get("poly_test_r2", row.get("poly_r2", 0.0))),
        "poly_degree": int(row.get("poly_degree", 1)),
        "base_score": float(row.get("base_score", 0.0)),
        "complexity": float(row.get("complexity", 0.0)),
        "composite_score": float(row.get("composite_score", 0.0)),
    }
    if "term_pair" in row:
        out["term_pair"] = pair
        out["mode"] = row.get("mode", "product")
    if row.get("mode") == "selected_step3_raw_linear":
        out["mode"] = row["mode"]
        out["pi_labels"] = tuple(row["pi_labels"])
        out["pi_formulas"] = tuple(row["pi_formulas"])
        out["pi_alphas"] = tuple(tuple(alpha) for alpha in row["pi_alphas"])
        out["feature_specs"] = tuple(dict(spec) for spec in row["feature_specs"])
        if row.get("regression_input"):
            out["regression_input"] = row["regression_input"]
    return out


def save_parameter_optimize_result(args, source_top_k, candidates, selected, result):
    path = parameter_result_cache_path(args)
    payload = {
        "metadata": parameter_result_cache_metadata(args, source_top_k),
        "candidates": candidates,
        "selected": list(selected),
        "result": result,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print("[Cache] Saved parameter optimize result to {0}".format(path))


def run_single_target(args=None):
    if args is None:
        args = parse_args()
    sample, case_data = load_case_from_settings(args)
    print_case_overview(sample, case_data)

    model, vocabs = load_prediction_model(args, sample)
    source_top_k = args.rank_source_top_k
    if source_top_k is None:
        source_top_k = max(args.top_k, args.independent_rank_top_n)
    else:
        source_top_k = max(int(source_top_k), args.independent_rank_top_n)

    top_k_start = time.perf_counter()
    raw_candidates = load_raw_candidates_cache(args, source_top_k)
    if raw_candidates is None:
        raw_candidates = evaluate_subset_candidates(
            model,
            sample,
            vocabs,
            top_n=max(source_top_k * 3, source_top_k),
            subset_size_penalty=args.subset_size_penalty,
            show_progress=True,
            progress_desc="Top-K base dimensionless numbers",
        )
        save_raw_candidates_cache(args, source_top_k, raw_candidates)
    top_k_elapsed = time.perf_counter() - top_k_start
    candidates, duplicate_candidates = deduplicate_candidates_by_alpha(raw_candidates)
    candidates = candidates[:source_top_k]
    if not candidates:
        raise RuntimeError("No dimensionless candidates were found.")
    print_top_candidates(candidates)
    print(
        "\n[Dedup] raw candidates={0}, duplicate alpha removed={1}, kept={2}".format(
            len(raw_candidates),
            len(duplicate_candidates),
            len(candidates),
        )
    )
    print("\n[Timing] Top-K base dimensionless numbers search: {:.3f} seconds".format(top_k_elapsed))

    log_pi_all, valid_pi = compute_log_pi_matrix(case_data["X"], candidates)
    target_all, valid_y, target_transform = select_target(
        case_data["Y"],
        args.output_index,
        args.target_transform,
    )
    valid = valid_pi & valid_y & np.all(np.isfinite(log_pi_all), axis=1) & np.isfinite(target_all)
    if np.count_nonzero(valid) < 5:
        raise RuntimeError("Too few valid rows after log-pi/target preprocessing.")

    log_pi = log_pi_all[valid]
    target = target_all[valid]
    print("\nTarget transform = {0}; valid rows = {1}/{2}".format(
        target_transform,
        np.count_nonzero(valid),
        len(valid),
    ))

    train_indices, test_indices = random_regression_split(
        len(target),
        train_fraction=0.8,
        random_seed=42,
    )
    log_pi_train = log_pi[train_indices]
    target_train = target[train_indices]
    log_pi_test = log_pi[test_indices]
    target_test = target[test_indices]
    print(
        "Regression split = 80% train / 20% test | train={0}, test={1}, seed=42".format(
            len(train_indices),
            len(test_indices),
        )
    )

    beta_opt_start = time.perf_counter()
    selected, skipped = select_independent_rank_candidates(
        candidates,
        max_rank=min(args.independent_rank_top_n, source_top_k),
    )
    print_independent_candidates(
        candidates,
        selected,
        skipped,
        min(args.independent_rank_top_n, source_top_k),
    )
    if not selected:
        raise RuntimeError("No linearly independent base pi group found in Rank 1-5.")

    a_matrix_selected = np.array([candidates[i]["full_alpha"] for i in selected], dtype=float)
    rank_a = int(np.linalg.matrix_rank(a_matrix_selected)) if len(selected) else 0
    use_accelerated_beta_search = rank_a >= int(args.accelerated_min_rank)

    exponents = parse_fraction_list(
        args.classic_exponents,
        [-3, -2, -1, Fraction(-1, 2), 0, Fraction(1, 2), 1, 2, 3],
    )
    print("\nClassic exponent set = {}".format([fraction_text(x) for x in exponents]))
    print("R2 threshold         = {:.4f}".format(args.r2_threshold))
    print("Beta search mode     = {}".format("accelerated" if use_accelerated_beta_search else "standard"))
    if use_accelerated_beta_search:
        effective_max_active = int(args.classic_max_active)
        if effective_max_active <= 0:
            effective_max_active = None
        if args.search_refine_top_m > 0:
            linear_retain_top_n = int(args.search_refine_top_m)
        else:
            linear_retain_top_n = int(args.accelerated_refine_top_m)
        print("classic max active   = {}".format(effective_max_active if effective_max_active else "none"))
        print("Step 3 refine top-M  = {}".format(linear_retain_top_n))
    else:
        effective_max_active = None
        linear_retain_top_n = None

    log_pi_selected = log_pi[:, selected]
    log_pi_selected_train = log_pi_train[:, selected]
    log_pi_selected_test = log_pi_test[:, selected]
    linear_rows = search_classic_exponent_products(
        log_pi_selected_train,
        target_train,
        log_pi_selected_test,
        target_test,
        exponents,
        show_progress=True,
        accelerated=use_accelerated_beta_search,
        max_active=effective_max_active,
        retain_top_n=linear_retain_top_n,
        chunk_size=args.accelerated_search_chunk_size,
        branch_bound_margin=args.branch_bound_margin,
    )
    if not linear_rows:
        raise RuntimeError("No valid classic-exponent product found.")

    print_search_results(
        "Step 2 | Linear/log-linear screening of total pi (ranked by test R2)",
        linear_rows,
        candidates,
        selected,
        sample,
        metric_key="linear_r2",
        max_print=5,
    )
    best_linear = linear_rows[0]

    poly_rows = refine_with_polynomial(
        linear_rows,
        target_train,
        target_test,
        max_degree=args.polynomial_max_degree,
        top_m=(linear_retain_top_n if use_accelerated_beta_search and args.search_refine_top_m <= 0 else args.search_refine_top_m),
        show_progress=True,
        log_pi_train=log_pi_selected_train,
        log_pi_test=log_pi_selected_test,
    )
    if use_accelerated_beta_search and args.search_refine_top_m <= 0:
        poly_refine_count = min(linear_retain_top_n, len(linear_rows))
    else:
        poly_refine_count = len(linear_rows) if args.search_refine_top_m <= 0 else min(args.search_refine_top_m, len(linear_rows))
    print("\n[Step 3] Polynomial refinement evaluated {0}/{1} beta candidates.".format(
        poly_refine_count,
        len(linear_rows),
    ))
    print_search_results(
        "Step 3 | Polynomial regression refinement (ranked by test R2)",
        poly_rows,
        candidates,
        selected,
        sample,
        metric_key="poly_r2",
        max_print=5,
    )
    best_poly = poly_rows[0] if poly_rows else None

    step3_best = best_poly if best_poly is not None else best_linear
    best_final = step3_best
    best_final_metric = "poly_r2" if best_poly is not None else "linear_r2"
    selected_step = 3
    step3_r2 = float(
        best_final.get(
            "poly_test_r2",
            best_final.get("linear_test_r2", best_final[best_final_metric]),
        )
    )

    two_pi_best = None
    if step3_r2 >= args.r2_threshold:
        direction_rows = [best_final]
        scales = collinear_scale_factors(exponents)
        print(
            "\n[Step 4] Step 3 test R2={0:.6f} reached threshold {1:.4f}; "
            "search collinear multiples of beta={2}, scales={3}.".format(
                step3_r2,
                args.r2_threshold,
                [fraction_text(x) for x in direction_rows[0]["beta"]],
                [fraction_text(x) for x in scales],
            )
        )
        raw_poly_rows = refine_with_raw_pi_polynomial(
            direction_rows,
            log_pi_selected_train,
            target_train,
            log_pi_selected_test,
            target_test,
            scales,
            max_degree=args.polynomial_max_degree,
            show_progress=True,
        )
        pi_train = pi_from_log_pi(log_pi_selected_train, direction_rows[0]["beta"])
        finite_pi_train = pi_train[np.isfinite(pi_train)]
        step4_notes = [
            "regression x     = Pi (not log Pi)",
            "ranking score = test R2 - {0:.2f} * complexity".format(BETA_COMPLEXITY_WEIGHT),
            "negative scales keep 1/Pi; direction is not normalized",
        ]
        if finite_pi_train.size:
            step4_notes.append(
                "Step 3 Pi train range = [{0:.6g}, {1:.6g}] (n={2})".format(
                    float(np.min(finite_pi_train)),
                    float(np.max(finite_pi_train)),
                    int(finite_pi_train.size),
                )
            )
        else:
            step4_notes.append("Step 3 Pi train range = unavailable")
        print_search_results(
            "Step 4 | Polynomial regression on Pi (not log Pi)",
            raw_poly_rows,
            candidates,
            selected,
            sample,
            metric_key="raw_poly_r2",
            max_print=5,
            header_notes=step4_notes,
        )
        if raw_poly_rows:
            best_step4 = raw_poly_rows[0]
            step3_score = beta_rank_score(
                best_final.get(
                    "poly_test_r2",
                    best_final.get("linear_test_r2", best_final[best_final_metric]),
                ),
                best_final["complexity"],
            )
            step4_score = beta_rank_score(
                best_step4["raw_poly_test_r2"],
                best_step4["complexity"],
            )
            print(
                "\n[Final] Step 3 rank score={0:.6f} vs Step 4 rank score={1:.6f}.".format(
                    step3_score,
                    step4_score,
                )
            )
            if step4_score > step3_score:
                best_final = best_step4
                best_final_metric = "raw_poly_r2"
                selected_step = 4
                print("[Final] Selected Step 4 (higher rank score).")
            else:
                selected_step = 3
                print("[Final] Selected Step 3 (rank score >= Step 4).")
        print("\nStep 5 | Two-pi search skipped because one-pi Step 3 reached threshold.")
    else:
        print(
            "\n[Step 4] Skipped; Step 3 test R2={0:.6f} is below threshold {1:.4f}. "
            "Two-pi search will be used.".format(step3_r2, args.r2_threshold)
        )

    if two_pi_best is None and step3_r2 < args.r2_threshold:
        two_pi_rows, _ = search_two_pi_models(
            candidates,
            selected,
            step3_best,
            log_pi_train,
            target_train,
            log_pi_test,
            target_test,
            args,
        )
        if two_pi_rows:
            two_pi_best = two_pi_rows[0]
            print_two_pi_result(two_pi_best, candidates)
        else:
            print_two_pi_result(None, candidates)

    print("\n" + "=" * 80)
    print("Final selected total dimensionless number")
    print("=" * 80)
    if two_pi_best is not None:
        print("selected step   = two_pi")
        if two_pi_best.get("mode") == "selected_step3_raw_linear":
            print("method          = two_pi_log_poly")
            print("regression x    = log(Pi)")
            print("train R2        = {:.6f}".format(two_pi_best.get("poly_train_r2", two_pi_best["linear_train_r2"])))
            print("test R2         = {:.6f}".format(two_pi_best.get("poly_test_r2", two_pi_best["linear_test_r2"])))
            print("poly degree     = {}".format(two_pi_best.get("poly_degree", 1)))
            print("selection       = raw linear on Pi")
            print("selection test R2 = {:.6f}".format(two_pi_best["linear_test_r2"]))
            print("threshold       = {:.6f}".format(args.r2_threshold))
            print("pi pair         = ({0}, {1})".format(*two_pi_best["pi_labels"]))
            for label, alpha, formula in zip(
                two_pi_best["pi_labels"],
                two_pi_best["pi_alphas"],
                two_pi_best["pi_formulas"],
            ):
                print("{0} alpha       = {1}".format(label, [float(x) for x in alpha]))
                print("{0} formula     = {1}".format(label, formula))
        elif "term_pair" in two_pi_best:
            print("method          = product_two_pi_poly")
            print("R2              = {:.6f}".format(two_pi_best["poly_r2"]))
            print("composite score = {:.6f}".format(two_pi_best["composite_score"]))
            print("complexity      = {:.6f}".format(two_pi_best["complexity"]))
            print("threshold       = {:.6f}".format(args.r2_threshold))
            left, right = two_pi_best["term_pair"]
            print("pi pair         = ({0}, {1})".format(term_label(left), term_label(right)))
            print("mode            = {}".format(two_pi_best.get("mode", "product")))
            for term in two_pi_best["term_pair"]:
                print("{0} alpha       = {1}".format(term_label(term), [float(x) for x in term_alpha(candidates, term)]))
                print("{0} formula     = {1}".format(term_label(term), term_formula(candidates, term)))
        else:
            print("method          = two_pi_poly")
            print("R2              = {:.6f}".format(two_pi_best["poly_r2"]))
            print("composite score = {:.6f}".format(two_pi_best["composite_score"]))
            print("complexity      = {:.6f}".format(two_pi_best["complexity"]))
            print("threshold       = {:.6f}".format(args.r2_threshold))
            i, j = two_pi_best["pair"]
            print("pi pair         = ({0}, {1})".format(candidate_label(i), candidate_label(j)))
            print("{0} alpha       = {1}".format(candidate_label(i), candidates[i]["full_alpha"]))
            print("{0} formula     = {1}".format(candidate_label(i), candidates[i]["formula"]))
            print("{0} alpha       = {1}".format(candidate_label(j), candidates[j]["full_alpha"]))
            print("{0} formula     = {1}".format(candidate_label(j), candidates[j]["formula"]))
        saved_result = {
            "kind": "two",
            "selected_step": "two_pi",
            "metric_key": "poly_r2",
            "row": minimal_two_pi_row(two_pi_best),
        }
    else:
        gamma = combined_alpha_for_search(candidates, selected, best_final["beta"])
        print("selected step   = {0}".format(selected_step))
        print("method          = {}".format(best_final_metric.replace("_r2", "")))
        if best_final_metric == "linear_r2":
            print("train R2        = {:.6f}".format(best_final["linear_train_r2"]))
            print("test R2         = {:.6f}".format(best_final["linear_test_r2"]))
        elif best_final_metric == "poly_r2":
            print("train R2        = {:.6f}".format(best_final["poly_train_r2"]))
            print("test R2         = {:.6f}".format(best_final["poly_test_r2"]))
        elif best_final_metric == "raw_poly_r2":
            print("train R2        = {:.6f}".format(best_final["raw_poly_train_r2"]))
            print("test R2         = {:.6f}".format(best_final["raw_poly_test_r2"]))
            print("regression x    = Pi (not log Pi)")
            if "beta_scale" in best_final:
                print("beta scale      = {}".format(fraction_text(best_final["beta_scale"])))
            print("poly degree     = {}".format(best_final["raw_poly_degree"]))
        else:
            print("R2              = {:.6f}".format(best_final[best_final_metric]))
        print("threshold       = {:.6f}".format(args.r2_threshold))
        print("complexity      = {:.6f}".format(best_final["complexity"]))
        if best_final_metric == "linear_r2":
            print(
                "rank score      = {:.6f}".format(
                    beta_rank_score(best_final["linear_test_r2"], best_final["complexity"])
                )
            )
        elif best_final_metric == "poly_r2":
            print(
                "rank score      = {:.6f}".format(
                    beta_rank_score(best_final["poly_test_r2"], best_final["complexity"])
                )
            )
        elif best_final_metric == "raw_poly_r2":
            print(
                "rank score      = {:.6f}".format(
                    beta_rank_score(best_final["raw_poly_test_r2"], best_final["complexity"])
                )
            )
        print("beta            = {}".format([fraction_text(x) for x in best_final["beta"]]))
        print("beta formula    = {}".format(total_formula_from_beta(candidates, selected, best_final["beta"])))
        print("combined alpha  = {}".format([fraction_text(x) for x in gamma]))
        print("combined formula= {}".format(alpha_to_formula(sample, gamma)))
        saved_result = {
            "kind": "one",
            "selected_step": selected_step,
            "metric_key": best_final_metric,
            "row": minimal_search_row(best_final, best_final_metric),
        }
    save_parameter_optimize_result(args, source_top_k, candidates, selected, saved_result)
    beta_opt_elapsed = time.perf_counter() - beta_opt_start
    print("\n[Timing] Beta optimization steps 1-4: {:.3f} seconds".format(beta_opt_elapsed))
    print("[Timing] Total measured time: {:.3f} seconds".format(top_k_elapsed + beta_opt_elapsed))


def run_logged_main(args):
    probe_args = copy.copy(args)
    _, case_data = load_case_from_settings(probe_args)
    indices = target_output_indices(args, case_data)

    for output_index in indices:
        target_args = copy.copy(args)
        target_args.output_index = int(output_index)
        target_args.output_label = output_column_label(case_data, int(output_index))
        log_path = output_log_path(target_args)
        with log_path.open("w", encoding="utf-8") as log_file:
            tee = TeeOutput(sys.stdout, log_file)
            with redirect_stdout(tee):
                print("[Log] Writing parameter_optimize output to {0}".format(log_path))
                if len(indices) > 1:
                    print(
                        "[Target] output_index={0}, output_label={1}".format(
                            target_args.output_index,
                            target_args.output_label,
                        )
                    )
                run_single_target(target_args)


if __name__ == "__main__":
    parsed_args = parse_args()
    run_logged_main(parsed_args)
