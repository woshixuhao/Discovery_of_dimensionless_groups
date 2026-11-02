"""Discover dimensionless dependence of variable RAD equation coefficients."""

import argparse
import ast
import copy
import csv
import itertools
import json
import pickle
import sys
from pathlib import Path

from rad_noise import add_noise_suffix, has_noise_tag, noise_tag, parse_noise_level

THIS_DIR = Path(__file__).resolve().parent

import numpy as np


import rad_knowledge as kal
import rad_optimization as po
from rad_knowledge import (
    alpha_complexity,
    alpha_to_formula,
    alpha_to_full_alpha,
    canonical_alpha,
    dimension_matrix,
    dimension_rank,
    make_subset_sample,
    model_log_prob_for_alpha,
    normalize_alpha_value,
    predict_distributions,
)


RAD_DIMENSIONLESS_DATASET_DIR = THIS_DIR / "data" / "dimensionless_dataset"
RAD_DISCOVERY_RESULTS_DIR = THIS_DIR / "discovery_results"
RAD_CASE_KEY = "rad"
RAD_CHECKPOINT_NAME = "knowledge_aware_learning_rad_best.pt"
NEAR_TIE_SCORE_ATOL = 1e-6
NEAR_TIE_SCORE_RTOL = 1e-9


def local_path(value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(THIS_DIR / path)


def configure_rad_args(args):
    args.case = RAD_CASE_KEY
    args.settings = local_path(args.settings)
    args.data_dir = local_path(args.data_dir)
    args.results_dir = local_path(args.results_dir)
    args.checkpoint_path = local_path(args.checkpoint_path)
    return args


def parse_args(argv=None):
    """Read local RAD settings; resolve relative paths beside this script."""
    settings_parser = argparse.ArgumentParser(add_help=False)
    settings_parser.add_argument("--settings", default=str(THIS_DIR / "settings.json"))
    known, _ = settings_parser.parse_known_args(argv)
    settings_path = Path(local_path(known.settings))
    with settings_path.open("r", encoding="utf-8") as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict):
        raise ValueError("RAD settings must be a JSON object.")

    parser = argparse.ArgumentParser(description="Discover RAD coefficient scaling using the local knowledge checkpoint.")
    parser.set_defaults(**settings)
    parser.add_argument("--settings", default=str(settings_path))
    parser.add_argument("--noise-level", dest="noise_level", default=settings.get("noise_level", 0.0))
    parser.add_argument("--data-dir", dest="data_dir", default=settings.get("data_dir", "data/dimensionless_dataset"))
    parser.add_argument("--out-dir", dest="results_dir", default=settings.get("results_dir", "discovery_results"))
    parser.add_argument("--checkpoint", dest="checkpoint_path", default=settings.get("checkpoint_path", RAD_CHECKPOINT_NAME))
    parser.add_argument("--device", default=settings.get("device"))
    parser.add_argument("--export-equations-only", action="store_true",
                        help="Export best-candidate equation TXT files from saved results without searching again.")
    args = parser.parse_args(argv)
    args.noise_level = parse_noise_level(args.noise_level)
    return configure_rad_args(args)


def rad_dimensionless_filenames(tag=""):
    return {
        "csv": add_noise_suffix("dimensionless_coefficients.csv", tag),
        "npz": add_noise_suffix("dimensionless_dataset.npz", tag),
        "equation": add_noise_suffix("equation_form.txt", tag),
        "cv_txt": add_noise_suffix("coefficient_cv_summary.txt", tag),
        "cv_csv": add_noise_suffix("coefficient_cv_summary.csv", tag),
    }


def load_rad_dimensionless_rows(dataset_dir=RAD_DIMENSIONLESS_DATASET_DIR, tag=""):
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / rad_dimensionless_filenames(tag)["csv"]
    if not csv_path.exists():
        raise FileNotFoundError("Dimensionless coefficient CSV not found: {0}".format(csv_path))

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise ValueError("No rows found in {0}".format(csv_path))

    return rows, fieldnames, csv_path


def load_rad_term_labels(dataset_dir=RAD_DIMENSIONLESS_DATASET_DIR, coef_columns=None, tag=""):
    dataset_dir = Path(dataset_dir)
    names = rad_dimensionless_filenames(tag)
    npz_path = dataset_dir / names["npz"]
    if npz_path.exists():
        payload = np.load(str(npz_path), allow_pickle=True)
        if "term_labels" in payload:
            labels = [str(value) for value in payload["term_labels"].tolist()]
            if labels:
                return labels

    equation_path = dataset_dir / names["equation"]
    if equation_path.exists():
        for line in equation_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("term_labels:"):
                try:
                    labels = ast.literal_eval(line.split(":", 1)[1].strip())
                    return [str(value) for value in labels]
                except Exception:
                    pass

    if coef_columns is None:
        return []
    return list(coef_columns)


def finite_float_column(rows, column):
    values = []
    for row in rows:
        try:
            value = float(row.get(column, "nan"))
        except Exception:
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype=float)


def pearson_corr(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(valid) < 3:
        return np.nan
    x_values = x_values[valid]
    y_values = y_values[valid]
    if np.std(x_values) < 1e-14 or np.std(y_values) < 1e-14:
        return np.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def analyze_rad_coefficient_cv(
    dataset_dir=RAD_DIMENSIONLESS_DATASET_DIR,
    cv_threshold=0.05,
    mean_eps=1e-12,
    tag="",
):
    """
    Analyze coefficient variability in data/dimensionless_dataset.

    A coefficient is treated as constant when std/abs(mean) <= cv_threshold.
    Otherwise it is treated as variable-related, and simple Pearson
    correlations with U, L, D, k, Pe, and Da are reported.
    """
    rows, fieldnames, csv_path = load_rad_dimensionless_rows(dataset_dir, tag=tag)
    coef_columns = [
        name for name in fieldnames
        if name.startswith("coef_")
    ]
    coef_columns = sorted(coef_columns, key=lambda name: int(name.split("_", 1)[1]))
    if not coef_columns:
        raise ValueError("No coef_* columns found in {0}".format(csv_path))

    term_labels = load_rad_term_labels(dataset_dir, coef_columns, tag=tag)
    variable_columns = [
        name for name in ["U", "L", "D", "k", "Pe", "Da"]
        if name in fieldnames
    ]

    results = []
    for index, coef_column in enumerate(coef_columns):
        values = finite_float_column(rows, coef_column)
        valid = np.isfinite(values)
        valid_values = values[valid]
        if len(valid_values) == 0:
            mean_value = np.nan
            std_value = np.nan
            cv_value = np.nan
        else:
            mean_value = float(np.mean(valid_values))
            std_value = float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0
            cv_value = float(std_value / max(abs(mean_value), mean_eps))

        classification = "constant" if np.isfinite(cv_value) and cv_value <= cv_threshold else "variable_related"

        correlations = {}
        for variable_column in variable_columns:
            variable_values = finite_float_column(rows, variable_column)
            correlations[variable_column] = pearson_corr(variable_values, values)

        finite_corr = {
            key: value for key, value in correlations.items()
            if np.isfinite(value)
        }
        if finite_corr:
            strongest_variable = max(finite_corr, key=lambda key: abs(finite_corr[key]))
            strongest_corr = finite_corr[strongest_variable]
        else:
            strongest_variable = ""
            strongest_corr = np.nan

        term_label = term_labels[index] if index < len(term_labels) else coef_column
        results.append(
            {
                "coef": coef_column,
                "term": term_label,
                "mean": mean_value,
                "std": std_value,
                "cv": cv_value,
                "classification": classification,
                "strongest_variable": strongest_variable,
                "strongest_corr": strongest_corr,
                "correlations": correlations,
            }
        )

    dataset_dir = Path(dataset_dir)
    names = rad_dimensionless_filenames(tag)
    summary_txt = dataset_dir / names["cv_txt"]
    summary_csv = dataset_dir / names["cv_csv"]

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames_out = [
            "coef",
            "term",
            "mean",
            "std",
            "cv",
            "classification",
            "strongest_variable",
            "strongest_corr",
        ] + ["corr_{0}".format(name) for name in variable_columns]
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        for item in results:
            row = {
                "coef": item["coef"],
                "term": item["term"],
                "mean": item["mean"],
                "std": item["std"],
                "cv": item["cv"],
                "classification": item["classification"],
                "strongest_variable": item["strongest_variable"],
                "strongest_corr": item["strongest_corr"],
            }
            for variable_column in variable_columns:
                row["corr_{0}".format(variable_column)] = item["correlations"].get(variable_column, np.nan)
            writer.writerow(row)

    lines = [
        "source: {0}".format(csv_path),
        "cv_threshold: {0}".format(cv_threshold),
        "classification rule: constant if std/abs(mean) <= cv_threshold",
        "",
        "coefficient CV statistics:",
    ]
    for item in results:
        lines.append(
            "{coef} ({term}) | mean={mean:.10e} | std={std:.10e} | cv={cv:.6g} | {classification} | strongest={strongest_variable} corr={strongest_corr:.6g}".format(
                **item
            )
        )
    lines.extend([
        "",
        "constant coefficients:",
        ", ".join(["{0} ({1})".format(item["coef"], item["term"]) for item in results if item["classification"] == "constant"]) or "None",
        "",
        "variable-related coefficients:",
        ", ".join(["{0} ({1})".format(item["coef"], item["term"]) for item in results if item["classification"] != "constant"]) or "None",
    ])
    summary_text = "\n".join(lines) + "\n"
    summary_txt.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print("[RAD] CV summary saved:")
    print(summary_txt)
    print(summary_csv)

    return results


def rad_dimensionless_sample():
    return {
        "id": "reaction_advection_diffusion_coefficients",
        "variables": [
            {
                "name": "characteristic velocity",
                "symbol": "U",
                "unit": "L/T",
                "dimension_vector": {"L": 1, "T": -1},
            },
            {
                "name": "characteristic length",
                "symbol": "L",
                "unit": "L",
                "dimension_vector": {"L": 1},
            },
            {
                "name": "diffusion coefficient",
                "symbol": "D",
                "unit": "L^2/T",
                "dimension_vector": {"L": 2, "T": -1},
            },
            {
                "name": "reaction rate",
                "symbol": "k",
                "unit": "1/T",
                "dimension_vector": {"T": -1},
            },
        ],
    }


def rad_dimensionless_search_args(output_label, base_args=None):
    if base_args is None:
        base_args = parse_args()
    args = copy.copy(base_args)
    args.output_index = 0
    args.output_label = output_label
    args.target_transform = "raw"
    return args


def load_rad_prediction_model(args, sample):
    """Load supplied inference weights without training data or a prior database."""
    return kal.load_checkpoint_model(args.checkpoint_path, device=args.device)


def rad_discover_candidates_with_knowledge_model(args, sample):
    """
    Use the trained knowledge-aware model to discover candidate dimensionless
    numbers from U, L, D, and k. No RAD-specific pi groups are hard-coded here.
    """
    model, vocabs = load_rad_prediction_model(args, sample)
    source_top_k = source_top_k_from_args(args)
    print("[RAD] Discovering candidate dimensionless numbers with the knowledge model.")
    print("[RAD] Candidate variables: {0}".format([v.get("symbol", "") for v in sample.get("variables", [])]))
    raw_candidates = evaluate_subset_candidates_model_only(
        model,
        sample,
        vocabs,
        subset_size_penalty=args.subset_size_penalty,
        show_progress=True,
        progress_desc="RAD knowledge-model dimensionless candidates",
    )
    candidates, duplicate_candidates = po.deduplicate_candidates_by_alpha(raw_candidates)
    candidates, near_tie_candidates = remove_near_tie_score_candidates(candidates)
    candidates = candidates[:source_top_k]
    if not candidates:
        raise RuntimeError("Knowledge model did not produce any dimensionless candidates for RAD variables.")
    po.print_top_candidates(candidates)
    print(
        "[RAD] raw candidates={0}, duplicate removed={1}, near-tie removed={2}, kept={3}".format(
            len(raw_candidates),
            len(duplicate_candidates),
            len(near_tie_candidates),
            len(candidates),
        )
    )
    return candidates, source_top_k


def rad_load_dimensionless_coefficient_dataset(dataset_dir=RAD_DIMENSIONLESS_DATASET_DIR, tag=""):
    rows, fieldnames, csv_path = load_rad_dimensionless_rows(dataset_dir, tag=tag)
    variable_columns = ["U", "L", "D", "k"]
    missing = [name for name in variable_columns if name not in fieldnames]
    if missing:
        raise ValueError("Missing variable columns in {0}: {1}".format(csv_path, missing))
    x = np.asarray(
        [[float(row[name]) for name in variable_columns] for row in rows],
        dtype=float,
    )
    case_names = [row.get("case", "case_{0:03d}".format(index)) for index, row in enumerate(rows)]
    return rows, fieldnames, csv_path, x, variable_columns, case_names


def safe_result_component(text):
    text = str(text)
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text) or "unknown"


def rad_target_for_dimensionless_search(coef_values, eps=1e-14):
    coef_values = np.asarray(coef_values, dtype=float)
    valid = np.isfinite(coef_values) & (np.abs(coef_values) > eps)
    signs = np.sign(coef_values[valid])
    if len(signs) == 0:
        raise ValueError("No finite nonzero coefficient values are available.")
    positive_count = int(np.count_nonzero(signs > 0))
    negative_count = int(np.count_nonzero(signs < 0))
    dominant_sign = 1.0 if positive_count >= negative_count else -1.0
    sign_consistent = (positive_count == 0) or (negative_count == 0)
    target = np.full(coef_values.shape, np.nan, dtype=float)
    target[valid] = np.log(np.abs(coef_values[valid]))
    return target, valid, {
        "dominant_sign": float(dominant_sign),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sign_consistent": bool(sign_consistent),
        "target_transform": "log(abs(coef))",
    }


def rad_write_dimensionless_discovery_result(
    path,
    coef_info,
    variable_columns,
    candidates,
    selected,
    saved_result,
    sign_info,
    valid_count,
):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("RAD coefficient dimensionless discovery summary\n")
        f.write("=" * 80 + "\n")
        f.write("coefficient: {0}\n".format(coef_info["coef"]))
        f.write("term: {0}\n".format(coef_info["term"]))
        f.write("classification: {0}\n".format(coef_info["classification"]))
        f.write("X: {0}\n".format(variable_columns))
        f.write("y: {0}\n".format(coef_info["coef"]))
        f.write("target_transform: {0}\n".format(sign_info["target_transform"]))
        f.write("dominant_sign: {0}\n".format(sign_info["dominant_sign"]))
        f.write("sign_consistent: {0}\n".format(sign_info["sign_consistent"]))
        f.write("positive_count: {0}\n".format(sign_info["positive_count"]))
        f.write("negative_count: {0}\n".format(sign_info["negative_count"]))
        f.write("valid_rows: {0}\n".format(valid_count))
        f.write("candidate_dimensionless_numbers:\n")
        for index, candidate in enumerate(candidates):
            f.write(
                "  {0}: alpha={1}, formula={2}\n".format(
                    po.candidate_label(index),
                    candidate["full_alpha"],
                    candidate["formula"],
                )
            )
        f.write("selected: {0}\n".format([po.candidate_label(index) for index in selected]))
        f.write("\nresult:\n")
        f.write("kind: {0}\n".format(saved_result["kind"]))
        f.write("row: {0}\n".format(saved_result["row"]))
        if saved_result["kind"] == "one":
            row = saved_result["row"]
            beta = row.get("beta")
            if beta is not None:
                f.write("beta_formula: {0}\n".format(po.total_formula_from_beta(candidates, selected, beta)))


def run_rad_dimensionless_discovery_for_variable_coefficients(
    args=None,
    dataset_dir=None,
    out_dir=None,
    cv_threshold=None,
    noise_level=None,
):
    """
    For each variable-related PDE coefficient, build X=[U,L,D,k], y=coef and
    run the local dimensionless search workflow.
    """
    if args is None:
        args = parse_args([])
    dataset_dir = args.data_dir if dataset_dir is None else dataset_dir
    out_dir = args.results_dir if out_dir is None else out_dir
    cv_threshold = args.cv_threshold if cv_threshold is None else cv_threshold
    noise_level = args.noise_level if noise_level is None else noise_level
    tag = noise_tag(parse_noise_level(noise_level))
    print("[RAD] noise_level={} tag={}".format(parse_noise_level(noise_level), tag or "clean"))
    cv_results = analyze_rad_coefficient_cv(dataset_dir=dataset_dir, cv_threshold=cv_threshold, tag=tag)
    variable_related = [
        item for item in cv_results
        if item.get("classification") == "variable_related"
    ]
    if not variable_related:
        print("[RAD] No variable-related coefficients were found by CV analysis.")
        return []

    rows, _, csv_path, x_all, variable_columns, case_names = rad_load_dimensionless_coefficient_dataset(
        dataset_dir,
        tag=tag,
    )
    sample = rad_dimensionless_sample()
    candidates, source_top_k = rad_discover_candidates_with_knowledge_model(args, sample)
    selected, _ = choose_independent_base_pi(args, sample, candidates, source_top_k)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for coef_info in variable_related:
        coef_column = coef_info["coef"]
        output_label = "{0}_{1}".format(coef_column, safe_result_component(coef_info["term"]))
        target_args = rad_dimensionless_search_args(output_label, args)
        coef_values = finite_float_column(rows, coef_column)
        target_all, valid_coef, sign_info = rad_target_for_dimensionless_search(coef_values)

        log_pi_all, valid_pi = po.compute_log_pi_matrix(x_all, candidates)
        valid = valid_coef & valid_pi & np.all(np.isfinite(log_pi_all), axis=1) & np.isfinite(target_all)
        if np.count_nonzero(valid) < 5:
            print("[RAD] Skip {0}: too few valid rows after preprocessing.".format(coef_column))
            continue

        log_pi = log_pi_all[valid]
        target = target_all[valid]
        x_valid = x_all[valid]
        y_valid = coef_values[valid]
        valid_case_names = [name for name, keep in zip(case_names, valid) if keep]

        npz_path = out_dir / add_noise_suffix("rad_dimensionless_dataset_{0}.npz".format(output_label), tag)
        np.savez_compressed(
            str(npz_path),
            X=x_valid,
            y=y_valid,
            target=target,
            variable_columns=np.asarray(variable_columns, dtype=object),
            case_names=np.asarray(valid_case_names, dtype=object),
            coef_column=coef_column,
            term=coef_info["term"],
            target_transform=sign_info["target_transform"],
        )

        log_path = out_dir / add_noise_suffix("rad_dimensionless_discovery_{0}.txt".format(output_label), tag)
        pkl_path = out_dir / add_noise_suffix("rad_dimensionless_discovery_{0}.pkl".format(output_label), tag)
        with log_path.open("w", encoding="utf-8") as log_file:
            tee = po.TeeOutput(sys.stdout, log_file)
            with po.redirect_stdout(tee):
                print("[RAD] Dimensionless discovery for {0} ({1})".format(coef_column, coef_info["term"]))
                print("[RAD] Source dataset = {0}".format(csv_path))
                print("[RAD] Saved coefficient dataset = {0}".format(npz_path))
                print("[RAD] X columns = {0}".format(variable_columns))
                print("[RAD] y column  = {0}".format(coef_column))
                print("[RAD] target   = {0}".format(sign_info["target_transform"]))
                print("[RAD] valid rows = {0}/{1}".format(np.count_nonzero(valid), len(valid)))
                print("[RAD] candidates = {0}".format([candidate["formula"] for candidate in candidates]))

                best_final, best_metric = search_one_pi(target_args, sample, candidates, selected, log_pi, target)
                saved_result = print_and_build_saved_result(
                    target_args,
                    sample,
                    candidates,
                    selected,
                    best_final,
                    best_metric,
                )
                rad_write_dimensionless_discovery_result(
                    log_file.name,
                    coef_info,
                    variable_columns,
                    candidates,
                    selected,
                    saved_result,
                    sign_info,
                    int(np.count_nonzero(valid)),
                )

        payload = {
            "coef_info": coef_info,
            "variable_columns": variable_columns,
            "sign_info": sign_info,
            "sample": sample,
            "candidates": candidates,
            "selected": selected,
            "result": saved_result,
            "dataset_npz": str(npz_path),
            "log_path": str(log_path),
        }
        with pkl_path.open("wb") as f:
            pickle.dump(payload, f)

        export_best_rank_equation(payload, out_dir, tag=tag)
        results.append(payload)
        print("[RAD] Saved dimensionless discovery result:")
        print(log_path)
        print(pkl_path)

    return results


def export_best_rank_equation(payload, results_dir, tag=""):
    """Export the final selected candidate, using the original training split.

    Keep the historical output directory/filename for compatibility; the TXT
    explicitly identifies the final selection rather than Step 2 Rank 2.
    """
    results_dir = Path(results_dir)
    info = payload["coef_info"]
    label = "{0}_{1}".format(info["coef"], safe_result_component(info["term"]))
    stem = add_noise_suffix(label, tag)
    dataset_path = results_dir / "rad_dimensionless_dataset_{}.npz".format(stem)
    with np.load(dataset_path, allow_pickle=True) as dataset:
        x = np.asarray(dataset["X"], dtype=float)
        y = np.asarray(dataset["y"], dtype=float).reshape(-1)
        target = np.asarray(dataset["target"], dtype=float).reshape(-1)

    saved = payload["result"]
    row = saved["row"]
    candidates, selected = payload["candidates"], payload["selected"]
    gamma = po.combined_alpha_for_search(candidates, selected, row["beta"])
    log_pi, valid = po.compute_log_pi_matrix(x, candidates)
    if not np.all(valid) or not np.all(np.isfinite(target)):
        raise ValueError("Saved search dataset contains invalid rows: {}".format(dataset_path))
    total = po.total_log_pi(log_pi[:, selected], row["beta"])
    train, test = po.random_regression_split(len(target), train_fraction=0.8, random_seed=42)
    metric = saved["metric_key"]
    degree = int(row.get("poly_degree", 1)) if metric == "poly_r2" else 1
    fit = po.fit_poly_1d_train_test(total[train], target[train], total[test], target[test], degree)
    if fit is None:
        raise ValueError("Selected dimensionless number is constant: {}".format(label))
    # Convert the standardized polynomial to powers of log(Pi).
    coordinate = np.polynomial.Polynomial([-fit["x_mean"] / fit["x_scale"], 1.0 / fit["x_scale"]])
    polynomial = np.polynomial.Polynomial(fit["coef"])(coordinate)
    expression = " ".join(
        "{:+.12g}{}".format(value, "" if power == 0 else "*log(Pi)^{}".format(power))
        for power, value in enumerate(polynomial.coef)
    )
    sign_info = payload["sign_info"]
    sign = float(sign_info["dominant_sign"])
    predicted = sign * np.exp(np.clip(fit["test_pred"], -700, 700))
    out_dir = results_dir / add_noise_suffix("rank2_linear_plots", tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "{}_rank2_linear_equation.txt".format(stem)
    lines = [
        "source_log: {}".format(results_dir / "rad_dimensionless_discovery_{}.txt".format(stem)),
        "source_dataset: {}".format(dataset_path),
        "coefficient: {}".format(info["coef"]),
        "term: {}".format(info["term"]),
        "selection: final selected best candidate (historical filename retained)",
        "metric: {}".format(metric),
        "combined_alpha: {}".format([po.fraction_text(value) for value in gamma]),
        "combined_form: {}".format(po.alpha_to_formula(payload["sample"], gamma)),
        "beta_formula: {}".format(po.total_formula_from_beta(candidates, selected, row["beta"])),
        "polynomial_degree: {}".format(degree),
        "log_equation: log(abs({})) = {}".format(info["coef"], expression),
        "equation: {} = {:+.0f} * exp({})".format(info["coef"], sign, expression),
        "sign_consistent: {}".format(sign_info["sign_consistent"]),
        "fit_split: 80% train / 20% test, seed=42; fit on training rows only",
        "r2_log_train: {:.12g}".format(fit["train_r2"]),
        "r2_log_test: {:.12g}".format(fit["test_r2"]),
        "r2_raw_test: {:.12g}".format(po.r2_score_np(y[test], predicted)),
    ]
    if not sign_info["sign_consistent"]:
        lines.append("sign_note: signed equation uses the majority sign; signs vary across cases")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[RAD] Best-candidate equation saved: {}".format(path))
    return path


def export_saved_best_rank_equations(results_dir, noise_level=0.0):
    """Regenerate equation TXT files from local saved search results."""
    results_dir = Path(results_dir)
    tag = noise_tag(parse_noise_level(noise_level))
    outputs = []
    for path in sorted(results_dir.glob("rad_dimensionless_discovery_coef_*.pkl")):
        if has_noise_tag(path.name, tag):
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            outputs.append(export_best_rank_equation(payload, results_dir, tag=tag))
    if not outputs:
        raise FileNotFoundError("No saved discovery results for this noise level in {}".format(results_dir))
    return outputs


def source_top_k_from_args(args):
    source_top_k = args.rank_source_top_k
    if source_top_k is None:
        source_top_k = max(args.top_k, args.independent_rank_top_n)
    else:
        source_top_k = max(int(source_top_k), args.independent_rank_top_n)
    return int(source_top_k)


def combination_count(n_value, k_value):
    if k_value < 0 or k_value > n_value:
        return 0
    k_value = min(k_value, n_value - k_value)
    numerator = 1
    denominator = 1
    for i in range(1, k_value + 1):
        numerator *= n_value - k_value + i
        denominator *= i
    return numerator // denominator


def constrained_decode_model_only(
    model,
    sample,
    vocabs,
    top_k_per_variable=5,
    top_m_candidates=10,
    complexity_penalty=0.05,
    probs=None,
):
    """
    Decode dimensionless candidates from the model top-k exponent space only.

    If the model top-k space has no valid dimensionless candidate, return an
    empty list. This intentionally does not call Buckingham-pi fallback.
    """
    if probs is None:
        probs = predict_distributions(model, sample, vocabs)
    d_matrix = dimension_matrix(sample)
    n_variables = len(sample["variables"])

    choices = []
    for i in range(n_variables):
        order = np.argsort(-probs[i])
        choices.append([kal.IDX_TO_ALPHA[int(idx)] for idx in order[:top_k_per_variable]])

    candidates = {}
    log_probs = np.log(probs + 1e-12)

    def try_add(alpha_tuple):
        alpha_arr = np.array(alpha_tuple, dtype=float)
        if np.allclose(alpha_arr, 0.0):
            return
        if not np.allclose(np.dot(d_matrix, alpha_arr), 0.0, atol=1e-8):
            return

        prior_score = 0.0
        for i, alpha_value in enumerate(alpha_tuple):
            normalized = normalize_alpha_value(alpha_value)
            if normalized not in kal.ALPHA_TO_IDX:
                return
            prior_score += log_probs[i, kal.ALPHA_TO_IDX[normalized]]

        complexity = alpha_complexity(alpha_tuple)
        score = prior_score - complexity_penalty * complexity
        key = canonical_alpha(alpha_tuple)
        if key not in candidates or score > candidates[key]["score"]:
            candidates[key] = {
                "score": score,
                "raw_alpha": tuple(normalize_alpha_value(x) for x in alpha_tuple),
                "canonical_alpha": key,
                "source": "model_subset",
            }

    for alpha_tuple in itertools.product(*choices):
        try_add(alpha_tuple)

    ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_m_candidates]


def evaluate_subset_candidates_model_only(
    model,
    sample,
    vocabs,
    min_subset_size=2,
    max_subset_size=None,
    top_k_per_variable=5,
    top_m_per_subset=3,
    subset_size_penalty=1.0,
    complexity_penalty=0.05,
    show_progress=False,
    progress_desc="Evaluating subsets",
):
    n_variables = len(sample["variables"])
    if max_subset_size is None:
        max_subset_size = n_variables
    max_subset_size = min(max_subset_size, n_variables)
    candidates = {}
    total_subsets = sum(
        combination_count(n_variables, k)
        for k in range(min_subset_size, max_subset_size + 1)
    )
    progress_bar = None
    processed_subsets = 0
    skipped_without_model_candidate = 0
    next_progress_report = 0.0

    if show_progress:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_subsets, desc=progress_desc, unit="subset")
        except ImportError:
            print("[Progress] {0}: 0/{1} subsets".format(progress_desc, total_subsets))

    for k in range(min_subset_size, max_subset_size + 1):
        for indices in itertools.combinations(range(n_variables), k):
            processed_subsets += 1
            subset = make_subset_sample(sample, indices)
            if dimension_rank(subset) == k:
                if progress_bar is not None:
                    progress_bar.update(1)
                elif show_progress:
                    fraction = processed_subsets / max(total_subsets, 1)
                    if fraction >= next_progress_report or processed_subsets == total_subsets:
                        print(
                            "[Progress] {0}: {1}/{2} subsets ({3:.1f}%)".format(
                                progress_desc,
                                processed_subsets,
                                total_subsets,
                                100 * fraction,
                            )
                        )
                        next_progress_report += 0.1
                continue

            probs = predict_distributions(model, subset, vocabs)
            decoded = constrained_decode_model_only(
                model,
                subset,
                vocabs,
                top_k_per_variable=top_k_per_variable,
                top_m_candidates=top_m_per_subset,
                complexity_penalty=complexity_penalty,
                probs=probs,
            )
            if not decoded:
                skipped_without_model_candidate += 1

            for cand in decoded:
                subset_alpha = cand["canonical_alpha"]
                if len(subset_alpha) != k:
                    continue

                score_alpha = cand.get("raw_alpha", subset_alpha)
                log_prob = model_log_prob_for_alpha(probs, score_alpha)
                if log_prob is None:
                    continue

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
                        "[Progress] {0}: {1}/{2} subsets ({3:.1f}%)".format(
                            progress_desc,
                            processed_subsets,
                            total_subsets,
                            100 * fraction,
                        )
                    )
                    next_progress_report += 0.1

    if progress_bar is not None:
        progress_bar.close()

    print("[Wall model] Model-only subset decode skipped {0} subsets with no valid model candidate.".format(
        skipped_without_model_candidate
    ))
    ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    return ranked


def remove_near_tie_score_candidates(candidates, abs_tol=NEAR_TIE_SCORE_ATOL, rel_tol=NEAR_TIE_SCORE_RTOL):
    kept = []
    removed = []
    last_kept_score = None
    for candidate in candidates:
        score = float(candidate.get("score", 0.0))
        if last_kept_score is not None and np.isclose(
            score,
            last_kept_score,
            atol=abs_tol,
            rtol=rel_tol,
        ):
            removed.append(candidate)
            continue
        kept.append(candidate)
        last_kept_score = score
    return kept, removed


def choose_independent_base_pi(args, sample, candidates, source_top_k):
    selected, skipped = po.select_independent_rank_candidates(
        candidates,
        max_rank=min(args.independent_rank_top_n, source_top_k),
    )
    print("\n[RAD] Selecting independent base dimensionless groups.")
    po.print_independent_candidates(
        candidates,
        selected,
        skipped,
        min(args.independent_rank_top_n, source_top_k),
    )
    if not selected:
        raise RuntimeError("No linearly independent RAD base pi group found.")
    return selected, None


def beta_search_settings(args, candidates, selected):
    a_matrix_selected = np.array([candidates[i]["full_alpha"] for i in selected], dtype=float)
    rank_a = int(np.linalg.matrix_rank(a_matrix_selected)) if len(selected) else 0
    use_accelerated_beta_search = rank_a >= int(args.accelerated_min_rank)

    exponents = po.parse_fraction_list(
        args.classic_exponents,
        [-3, -2, -1, po.Fraction(-1, 2), 0, po.Fraction(1, 2), 1, 2, 3],
    )
    print("\nClassic exponent set = {}".format([po.fraction_text(x) for x in exponents]))
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

    return exponents, use_accelerated_beta_search, effective_max_active, linear_retain_top_n


def search_one_pi(args, sample, candidates, selected, log_pi, target):
    exponents, accelerated, max_active, retain_top_n = beta_search_settings(args, candidates, selected)
    log_pi_selected = log_pi[:, selected]
    train_indices, test_indices = po.random_regression_split(
        len(target),
        train_fraction=0.8,
        random_seed=42,
    )
    log_pi_selected_train = log_pi_selected[train_indices]
    target_train = target[train_indices]
    log_pi_selected_test = log_pi_selected[test_indices]
    target_test = target[test_indices]
    print(
        "Regression split = 80% train / 20% test | train={0}, test={1}, seed=42".format(
            len(train_indices),
            len(test_indices),
        )
    )
    linear_rows = po.search_classic_exponent_products(
        log_pi_selected_train,
        target_train,
        log_pi_selected_test,
        target_test,
        exponents,
        show_progress=True,
        accelerated=accelerated,
        max_active=max_active,
        retain_top_n=retain_top_n,
        chunk_size=args.accelerated_search_chunk_size,
        branch_bound_margin=args.branch_bound_margin,
    )
    if not linear_rows:
        raise RuntimeError("No valid classic-exponent product found.")

    linear_rows.sort(key=lambda row: po.beta_sort_key(row, "linear_test_r2"))
    po.print_search_results(
        "Step 2 | Linear/log-linear screening of total pi (ranked by test R2)",
        linear_rows,
        candidates,
        selected,
        sample,
        metric_key="linear_r2",
        max_print=5,
    )
    best_linear = linear_rows[0]

    poly_top_m = retain_top_n if accelerated and args.search_refine_top_m <= 0 else args.search_refine_top_m
    poly_rows = po.refine_with_polynomial(
        linear_rows,
        target_train,
        target_test,
        max_degree=args.polynomial_max_degree,
        top_m=poly_top_m,
        show_progress=True,
        log_pi_train=log_pi_selected_train,
        log_pi_test=log_pi_selected_test,
    )
    if accelerated and args.search_refine_top_m <= 0:
        poly_refine_count = min(retain_top_n, len(linear_rows))
    else:
        poly_refine_count = len(linear_rows) if args.search_refine_top_m <= 0 else min(args.search_refine_top_m, len(linear_rows))
    print("\n[Step 3] Polynomial refinement evaluated {0}/{1} beta candidates.".format(
        poly_refine_count,
        len(linear_rows),
    ))
    poly_rows.sort(key=lambda row: po.beta_sort_key(row, "poly_test_r2"))
    po.print_search_results(
        "Step 3 | Polynomial regression refinement (ranked by test R2)",
        poly_rows,
        candidates,
        selected,
        sample,
        metric_key="poly_r2",
        max_print=5,
    )

    best_poly = poly_rows[0] if poly_rows else None
    best_final = best_poly if best_poly is not None else best_linear
    best_metric = "poly_r2" if best_poly is not None else "linear_r2"

    return best_final, best_metric


def print_and_build_saved_result(args, sample, candidates, selected, best_final, best_metric):
    print("\n" + "=" * 80)
    print("Final selected total dimensionless number")
    print("=" * 80)
    print("selection rule  = best one-pi polynomial/log-linear candidate")
    gamma = po.combined_alpha_for_search(candidates, selected, best_final["beta"])
    print("method          = {}".format(best_metric.replace("_r2", "")))
    if best_metric == "linear_r2":
        print("train R2        = {:.6f}".format(best_final["linear_train_r2"]))
        print("test R2         = {:.6f}".format(best_final["linear_test_r2"]))
    elif best_metric == "poly_r2":
        print("train R2        = {:.6f}".format(best_final["poly_train_r2"]))
        print("test R2         = {:.6f}".format(best_final["poly_test_r2"]))
    else:
        print("R2              = {:.6f}".format(best_final[best_metric]))
    print("threshold       = {:.6f}".format(args.r2_threshold))
    print("complexity      = {:.6f}".format(best_final["complexity"]))
    if best_metric == "linear_r2":
        print(
            "rank score      = {:.6f}".format(
                po.beta_rank_score(best_final["linear_test_r2"], best_final["complexity"])
            )
        )
    elif best_metric == "poly_r2":
        print(
            "rank score      = {:.6f}".format(
                po.beta_rank_score(best_final["poly_test_r2"], best_final["complexity"])
            )
        )
    print("beta            = {}".format([po.fraction_text(x) for x in best_final["beta"]]))
    print("beta formula    = {}".format(po.total_formula_from_beta(candidates, selected, best_final["beta"])))
    print("combined alpha  = {}".format([po.fraction_text(x) for x in gamma]))
    print("combined formula= {}".format(po.alpha_to_formula(sample, gamma)))
    return {
        "kind": "one",
        "metric_key": best_metric,
        "row": po.minimal_search_row(best_final, best_metric),
    }


if __name__ == "__main__":
    args = parse_args()
    if args.export_equations_only:
        export_saved_best_rank_equations(args.results_dir, noise_level=args.noise_level)
    else:
        run_rad_dimensionless_discovery_for_variable_coefficients(args=args)
