"""Local discrete-exponent search, regression, scoring and logging for RAD."""

import heapq
import itertools
from contextlib import redirect_stdout
from fractions import Fraction

import numpy as np

from rad_knowledge import alpha_to_formula

BETA_COMPLEXITY_WEIGHT = 0.01


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
