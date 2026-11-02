import argparse
import ast
from collections import Counter
import csv
import glob
import os
import random

import numpy as np
import torch

from GGA import GGA
from rad_noise import (
    add_noise_argument,
    add_noise_suffix,
    model_best_regex,
    noise_tag,
    parse_noise_level,
)
from train_surrogate_model import NN


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_DATA_DIR = os.path.join(ROOT_DIR, "data", "cases")
# The uploaded package may ship case files directly in cases/.
if not os.path.isdir(CASE_DATA_DIR):
    CASE_DATA_DIR = os.path.join(ROOT_DIR, "cases")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
METADATA_DIR = os.path.join(ROOT_DIR, "data", "discovery_metadata")
RESULT_DIR = os.path.join(ROOT_DIR, "discovery_results")
TEMP_SAVE_DIR = os.path.join(ROOT_DIR, "temp_save")
DIMENSIONLESS_DATASET_DIR = os.path.join(ROOT_DIR, "data", "dimensionless_dataset")
NOISE_LEVEL = 0.0
NOISE_TAG = ""


def configure_noise_paths(noise_level=0.0):
    global NOISE_LEVEL, NOISE_TAG
    NOISE_LEVEL = parse_noise_level(noise_level)
    NOISE_TAG = noise_tag(NOISE_LEVEL)
    return NOISE_TAG


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_text):
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested, but torch.cuda.is_available() is False.")
    return device


def normalize_case_name(case_name):
    case_name = str(case_name).strip()
    if case_name.endswith(".npz"):
        case_name = os.path.splitext(case_name)[0]
    if case_name.endswith("_best"):
        case_name = case_name[:-5]
    return case_name


def case_paths(case_name):
    case_name = normalize_case_name(case_name)
    data_path = os.path.join(CASE_DATA_DIR, "{}.npz".format(case_name))
    model_path = os.path.join(MODEL_DIR, "{}_best.pt".format(case_name))
    return case_name, data_path, model_path


def find_trained_cases(tag=None):
    if tag is None:
        tag = NOISE_TAG
    pattern = os.path.join(MODEL_DIR, "case_*_best.pt")
    matcher = model_best_regex(tag)
    model_paths = sorted(glob.glob(pattern))
    case_names = []
    for model_path in model_paths:
        basename = os.path.basename(model_path)
        if not matcher.match(basename):
            continue
        name = os.path.splitext(basename)[0]
        case_names.append(normalize_case_name(name))
    return case_names


def torch_load_checkpoint(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_case_domain(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError("Case data file not found: {}".format(data_path))
    payload = np.load(data_path)
    x = np.asarray(payload["x"], dtype=np.float32)
    t = np.asarray(payload["t"], dtype=np.float32)
    variables = {}
    for key in ["U", "L", "D", "k", "Pe", "Da"]:
        if key in payload:
            variables[key] = float(np.asarray(payload[key]))
    return x, t, variables


def build_model_from_checkpoint(checkpoint, device):
    config = checkpoint.get("model_config", {})
    hidden = int(config.get("Neurons_Per_Layer", config.get("hidden_neuron", 64)))
    layers = int(config.get("Num_Hidden_Layers", config.get("num_hidden_layers", 4)))
    activation = config.get("Activation_Function", "Tanh")

    model = NN(
        Num_Hidden_Layers=layers,
        Neurons_Per_Layer=hidden,
        Input_Dim=2,
        Output_Dim=1,
        Data_Type=torch.float32,
        Device=device,
        Activation_Function=activation,
        Batch_Norm=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def grad_component(output, inputs, component_index, create_graph=True):
    grad = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad[:, component_index:component_index + 1]


def predict_with_derivatives(model, x_domain, t_domain, grid_size, device):
    # x_grid = np.linspace(float(np.min(x_domain)), float(np.max(x_domain)), grid_size).astype(np.float32)
    # t_grid = np.linspace(float(np.min(t_domain)), float(np.max(t_domain)), grid_size).astype(np.float32)
    x_grid = np.linspace(0.1,0.9, grid_size).astype(np.float32)
    t_grid = np.linspace(0.1,0.9, grid_size).astype(np.float32)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid, indexing="ij")

    inputs_np = np.column_stack([X_grid.reshape(-1), T_grid.reshape(-1)]).astype(np.float32)
    inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device, requires_grad=True)

    u = model(inputs)
    u_x = grad_component(u, inputs, 0, create_graph=True)
    u_t = grad_component(u, inputs, 1, create_graph=True)
    u_xx = grad_component(u_x, inputs, 0, create_graph=True)
    u_tt = grad_component(u_t, inputs, 1, create_graph=True)
    u_xxx = grad_component(u_xx, inputs, 0, create_graph=True)

    nx = grid_size
    nt = grid_size
    metadata = {
        "x": x_grid,
        "t": t_grid,
        "u": u.detach().cpu().numpy().reshape(nx * nt, 1),
        "u_t": u_t.detach().cpu().numpy().reshape(nx * nt, 1),
        "u_tt": u_tt.detach().cpu().numpy().reshape(nx * nt, 1),
        "u_x": u_x.detach().cpu().numpy().reshape(nx * nt, 1),
        "u_xx": u_xx.detach().cpu().numpy().reshape(nx * nt, 1),
        "u_xxx": u_xxx.detach().cpu().numpy().reshape(nx * nt, 1),
    }
    return metadata


def save_metadata(case_name, metadata, variables):
    os.makedirs(METADATA_DIR, exist_ok=True)
    path = os.path.join(METADATA_DIR, "{}_ann_metadata.npz".format(case_name))
    np.savez_compressed(
        path,
        x=metadata["x"],
        t=metadata["t"],
        u=metadata["u"],
        u_t=metadata["u_t"],
        u_tt=metadata["u_tt"],
        u_x=metadata["u_x"],
        u_xx=metadata["u_xx"],
        u_xxx=metadata["u_xxx"],
        variables=np.array([variables], dtype=object),
    )
    return path


def module_to_text(module):
    counts = {}
    labels = {
        0: "u",
        1: "u_x",
        2: "u_xx",
        3: "u_xxx",
    }
    for item in module:
        counts[item] = counts.get(item, 0) + 1
    parts = []
    for key in sorted(counts.keys()):
        base = labels.get(key, "unknown")
        power = counts[key]
        if power == 1:
            parts.append(base)
        else:
            parts.append("{}^{}".format(base, power))
    if not parts:
        return "1"
    return " ".join(parts)


def term_to_text(module, diff_module):
    base = module_to_text(module)
    diff_x = int(diff_module[1])
    if diff_x <= 0:
        return base
    if diff_x == 1:
        return "d/dx({})".format(base)
    return "d^{} / dx^{}({})".format(diff_x, diff_x, base)


def equation_to_text(target_name, gene, gene_diff, coef):
    terms = []
    coef = np.asarray(coef, dtype=float).reshape(-1)
    for index, module in enumerate(gene):
        if index >= len(coef):
            break
        if index < len(gene_diff):
            diff_module = gene_diff[index]
        else:
            diff_module = [0, 0]
        term_text = term_to_text(module, diff_module)
        terms.append("{:+.6e}*{}".format(float(coef[index]), term_text))
    rhs = " ".join(terms)
    if rhs.startswith("+"):
        rhs = rhs[1:].strip()
    return "{} = {}".format(target_name, rhs)


def delete_boundary_flat(values, nx, nt, delete_num):
    values_2d = np.asarray(values, dtype=float).reshape(nx, nt)
    if delete_num <= 0:
        return values_2d.reshape(nx * nt, 1)
    trim = int(delete_num // 2)
    if trim <= 0:
        return values_2d.reshape(nx * nt, 1)
    values_del = values_2d[trim:nx - trim, trim:nt - trim]
    return values_del.reshape((nx - 2 * trim) * (nt - 2 * trim), 1)


def evaluate_correct_equation(metadata, variables, epi=1e-6, delete_num=10):
    """
    Evaluate the known RAD equation library [u, u^2, u_xx, u_x].

    The regression solves:
        u_t = a*u + b*u^2 + c*u_xx + d*u_x
    """
    x = metadata["x"]
    t = metadata["t"]
    nx = int(len(x))
    nt = int(len(t))

    u = delete_boundary_flat(metadata["u"], nx, nt, delete_num)
    u_t = delete_boundary_flat(metadata["u_t"], nx, nt, delete_num)
    u_x = delete_boundary_flat(metadata["u_x"], nx, nt, delete_num)
    u_xx = delete_boundary_flat(metadata["u_xx"], nx, nt, delete_num)

    theta = np.hstack((u, u ** 2, u_xx, u_x))
    coef, _, _, _ = np.linalg.lstsq(theta, u_t, rcond=None)
    residual = u_t - np.dot(theta, coef)
    mse = float(np.mean(residual ** 2))

    # Complexity is counted consistently with the four symbolic terms:
    # len([u]) + len([u, u]) + len([u_xx]) + len([u_x]).
    length_penalty_coef = 5
    fitness = mse + float(epi) * length_penalty_coef

    term_names = ["u", "u^2", "u_xx", "u_x"]
    pieces = []
    for value, name in zip(coef.reshape(-1), term_names):
        pieces.append("{:+.6e}*{}".format(float(value), name))
    rhs = " ".join(pieces)
    if rhs.startswith("+"):
        rhs = rhs[1:].strip()
    equation_text = "u_t = {}".format(rhs)

    true_equation_text = None
    if variables and "Da" in variables and "Pe" in variables:
        da = float(variables["Da"])
        pe = float(variables["Pe"])
        true_equation_text = (
            "u_t = {:+.6e}*u {:+.6e}*u^2 {:+.6e}*u_xx {:+.6e}*u_x"
        ).format(da, -da, 1.0 / pe, -1.0)
        if true_equation_text.startswith("u_t = +"):
            true_equation_text = "u_t = " + true_equation_text[len("u_t = +"):]

    return {
        "terms": term_names,
        "coef": coef.reshape(-1).tolist(),
        "mse": mse,
        "fitness": float(fitness),
        "length_penalty_coef": length_penalty_coef,
        "equation_text": equation_text,
        "true_equation_text": true_equation_text,
    }


def run_gga(case_name, metadata, args):
    os.makedirs(TEMP_SAVE_DIR, exist_ok=True)
    previous_cwd = os.getcwd()
    os.chdir(ROOT_DIR)
    try:
        ga = GGA(
            metadata["x"],
            metadata["t"],
            metadata["u"],
            metadata["u_x"],
            metadata["u_xx"],
            metadata["u_xxx"],
            metadata["u_t"],
            metadata["u_tt"],
            args.epi,
            dim=1,
            delete_num=args.delete_num,
            creterion=args.criterion,
            temp_tag=NOISE_TAG,
        )
        ga.pop_size = args.pop_size
        ga.n_generations = args.n_generations
        ga.max_length = args.max_length
        best_gene, _, best_coef, best_fitness = ga.evolution()
        best_gene_diff = ga.Chrom_diff[0]
        if hasattr(ga, "name") and len(ga.name):
            target_name = ga.name[0]
        else:
            target_name = "u_t"
    finally:
        os.chdir(previous_cwd)

    equation_text = equation_to_text(target_name, best_gene, best_gene_diff, best_coef)
    result = {
        "case_name": case_name,
        "target_name": target_name,
        "best_gene": best_gene,
        "best_gene_diff": best_gene_diff,
        "best_coef": np.asarray(best_coef).reshape(-1).tolist(),
        "best_fitness": float(best_fitness),
        "equation_text": equation_text,
    }
    return result


def format_result_block(case_name, result, metadata_path, variables):
    lines = [
        "case: {}".format(case_name),
        "metadata: {}".format(metadata_path),
    ]
    if variables:
        lines.append("variables: {}".format(variables))
    lines.extend(
        [
            "fitness: {:.10e}".format(result["best_fitness"]),
            "equation:",
            result["equation_text"],
            "gene: {}".format(result["best_gene"]),
            "gene_diff: {}".format(result["best_gene_diff"]),
            "coef: {}".format(result["best_coef"]),
        ]
    )
    return "\n".join(lines) + "\n"


def save_result(case_name, result, metadata_path, variables, correct_result=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    txt_path = os.path.join(RESULT_DIR, "{}_discovered_equation.txt".format(case_name))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_result_block(case_name, result, metadata_path, variables))
        if correct_result is not None:
            f.write("\ncorrect_library_check:\n")
            if correct_result.get("true_equation_text"):
                f.write("true equation: {}\n".format(correct_result["true_equation_text"]))
            f.write("regressed equation: {}\n".format(correct_result["equation_text"]))
            f.write("mse: {:.10e}\n".format(correct_result["mse"]))
            f.write("fitness: {:.10e}\n".format(correct_result["fitness"]))
            f.write("terms: {}\n".format(correct_result["terms"]))
            f.write("coef: {}\n".format(correct_result["coef"]))
    return txt_path


def discover_one_case(case_name, args, device, run_correct_check=True):
    case_name, data_path, model_path = case_paths(case_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found: {}".format(model_path))

    x_domain, t_domain, variables = load_case_domain(data_path)
    checkpoint = torch_load_checkpoint(model_path, device)
    model, model_config = build_model_from_checkpoint(checkpoint, device)

    print("Case: {}".format(case_name))
    print("Use model: {}".format(model_path))
    print("Use device: {}".format(device))
    print("Model config: {}".format(model_config))
    print(
        "Uniform discovery grid: nx={} nt={} x=[{:.6g}, {:.6g}] t=[{:.6g}, {:.6g}]".format(
            args.grid_size,
            args.grid_size,
            float(np.min(x_domain)),
            float(np.max(x_domain)),
            float(np.min(t_domain)),
            float(np.max(t_domain)),
        )
    )

    metadata = predict_with_derivatives(model, x_domain, t_domain, args.grid_size, device)
    metadata_path = save_metadata(case_name, metadata, variables)
    print("ANN metadata saved: {}".format(metadata_path))

    correct_result = None
    if run_correct_check:
        correct_result = evaluate_correct_equation(
            metadata,
            variables,
            epi=args.epi,
            delete_num=args.delete_num,
        )

    result = run_gga(case_name, metadata, args)

    del model
    del checkpoint
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    return result, metadata_path, variables, correct_result


def run_all(args, device):
    case_names = find_trained_cases()
    if not case_names:
        raise FileNotFoundError("No trained case_*_best.pt models found in {}".format(MODEL_DIR))

    os.makedirs(RESULT_DIR, exist_ok=True)
    all_result_path = os.path.join(RESULT_DIR, add_noise_suffix(args.all_results_file, NOISE_TAG))
    print("Run all cases: {}".format(len(case_names)))
    print("All results will be saved to {}".format(all_result_path))

    with open(all_result_path, "w", encoding="utf-8") as f:
        for index, case_name in enumerate(case_names):
            print("\n[{}/{}] Running GGA for {}".format(index + 1, len(case_names), case_name))
            result, metadata_path, variables, _ = discover_one_case(
                case_name,
                args,
                device,
                run_correct_check=False,
            )
            f.write(format_result_block(case_name, result, metadata_path, variables))
            if index != len(case_names) - 1:
                f.write("\n")
            f.flush()
            print("Saved {} result to all-results file.".format(case_name))

    return all_result_path


def canonical_term_key(module, diff_module):
    module_key = tuple(sorted(int(item) for item in module))
    diff_key = tuple(int(item) for item in diff_module)
    return module_key, diff_key


def canonical_equation_key(gene, gene_diff):
    terms = []
    for index, module in enumerate(gene):
        if index < len(gene_diff):
            diff_module = gene_diff[index]
        else:
            diff_module = [0, 0]
        terms.append(canonical_term_key(module, diff_module))
    return tuple(sorted(terms))


def canonical_equation_text(equation_key):
    terms = []
    for module_key, diff_key in equation_key:
        terms.append(term_to_text(list(module_key), list(diff_key)))
    return "u_t = " + " + ".join(terms)


def read_discovery_result_blocks(results_path):
    if not os.path.exists(results_path):
        raise FileNotFoundError("Discovery result file not found: {}".format(results_path))

    blocks = []
    current = {}
    with open(results_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("case:"):
            if current:
                blocks.append(current)
            current = {"case": line.split(":", 1)[1].strip()}
        elif line.startswith("metadata:"):
            current["metadata"] = line.split(":", 1)[1].strip()
        elif line.startswith("variables:"):
            current["variables"] = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("gene:"):
            current["gene"] = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("gene_diff:"):
            current["gene_diff"] = ast.literal_eval(line.split(":", 1)[1].strip())
        index += 1

    if current:
        blocks.append(current)
    return blocks


def summarize_equation_form_frequencies(results_path=None, save_path=None):
    """
    Read discovered equations and count equation-form frequencies.

    Term order does not affect the statistics. The most frequent equation form
    is treated as the correct equation form.
    """
    if results_path is None:
        results_path = os.path.join(RESULT_DIR, add_noise_suffix("all_discovered_equations.txt", NOISE_TAG))

    blocks = read_discovery_result_blocks(results_path)
    counter = Counter()
    cases_by_key = {}
    for block in blocks:
        if "gene" not in block or "gene_diff" not in block:
            continue
        key = canonical_equation_key(block["gene"], block["gene_diff"])
        counter[key] += 1
        cases_by_key.setdefault(key, []).append(block.get("case", "unknown_case"))

    if not counter:
        raise ValueError("No valid gene/gene_diff records found in {}".format(results_path))

    ranked = counter.most_common()
    best_key, best_count = ranked[0]
    total = sum(counter.values())

    lines = [
        "source: {}".format(results_path),
        "total_cases: {}".format(total),
        "unique_equation_forms: {}".format(len(ranked)),
        "",
        "most_frequent_equation_form:",
        "frequency: {}/{}".format(best_count, total),
        "equation_form: {}".format(canonical_equation_text(best_key)),
        "cases: {}".format(cases_by_key[best_key]),
        "",
        "all_equation_form_frequencies:",
    ]
    for rank, (key, count) in enumerate(ranked, start=1):
        lines.append(
            "{}. frequency: {}/{} | equation_form: {} | cases: {}".format(
                rank,
                count,
                total,
                canonical_equation_text(key),
                cases_by_key[key],
            )
        )

    summary_text = "\n".join(lines) + "\n"
    print("\nEquation-form frequency summary:")
    print(summary_text)

    if save_path is None:
        save_path = os.path.join(
            RESULT_DIR,
            add_noise_suffix("equation_form_frequency_summary.txt", NOISE_TAG),
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return {
        "results_path": results_path,
        "summary_path": save_path,
        "total_cases": total,
        "unique_equation_forms": len(ranked),
        "best_frequency": best_count,
        "best_key": best_key,
        "best_equation_form": canonical_equation_text(best_key),
        "counter": counter,
        "cases_by_key": cases_by_key,
    }


def finite_diff_x_array(values_2d, x_values, derivative_order):
    values_2d = np.asarray(values_2d, dtype=float)
    derivative_order = int(derivative_order)
    if derivative_order <= 0:
        return values_2d

    dx = float(x_values[1] - x_values[0])
    nx, nt = values_2d.shape
    out = np.zeros((nx, nt), dtype=float)

    if derivative_order == 1:
        out[1:nx - 1, :] = (values_2d[2:nx, :] - values_2d[0:nx - 2, :]) / (2.0 * dx)
        out[0, :] = (-1.5 * values_2d[0, :] + 2.0 * values_2d[1, :] - 0.5 * values_2d[2, :]) / dx
        out[nx - 1, :] = (
            1.5 * values_2d[nx - 1, :] - 2.0 * values_2d[nx - 2, :] + 0.5 * values_2d[nx - 3, :]
        ) / dx
        return out

    if derivative_order == 2:
        out[1:nx - 1, :] = (
            values_2d[2:nx, :] - 2.0 * values_2d[1:nx - 1, :] + values_2d[0:nx - 2, :]
        ) / (dx ** 2)
        out[0, :] = (
            2.0 * values_2d[0, :] - 5.0 * values_2d[1, :] + 4.0 * values_2d[2, :] - values_2d[3, :]
        ) / (dx ** 2)
        out[nx - 1, :] = (
            2.0 * values_2d[nx - 1, :]
            - 5.0 * values_2d[nx - 2, :]
            + 4.0 * values_2d[nx - 3, :]
            - values_2d[nx - 4, :]
        ) / (dx ** 2)
        return out

    if derivative_order == 3:
        out[2:nx - 2, :] = (
            0.5 * values_2d[4:nx, :]
            - values_2d[3:nx - 1, :]
            + values_2d[1:nx - 3, :]
            - 0.5 * values_2d[0:nx - 4, :]
        ) / (dx ** 3)
        out[0, :] = (
            -2.5 * values_2d[0, :]
            + 9.0 * values_2d[1, :]
            - 12.0 * values_2d[2, :]
            + 7.0 * values_2d[3, :]
            - 1.5 * values_2d[4, :]
        ) / (dx ** 3)
        out[1, :] = (
            -2.5 * values_2d[1, :]
            + 9.0 * values_2d[2, :]
            - 12.0 * values_2d[3, :]
            + 7.0 * values_2d[4, :]
            - 1.5 * values_2d[5, :]
        ) / (dx ** 3)
        out[nx - 1, :] = (
            2.5 * values_2d[nx - 1, :]
            - 9.0 * values_2d[nx - 2, :]
            + 12.0 * values_2d[nx - 3, :]
            - 7.0 * values_2d[nx - 4, :]
            + 1.5 * values_2d[nx - 5, :]
        ) / (dx ** 3)
        out[nx - 2, :] = (
            2.5 * values_2d[nx - 2, :]
            - 9.0 * values_2d[nx - 3, :]
            + 12.0 * values_2d[nx - 4, :]
            - 7.0 * values_2d[nx - 5, :]
            + 1.5 * values_2d[nx - 6, :]
        ) / (dx ** 3)
        return out

    return finite_diff_x_array(finite_diff_x_array(values_2d, x_values, 3), x_values, derivative_order - 3)


def metadata_term_array(metadata, term_key):
    module_key, diff_key = term_key
    x_values = metadata["x"]
    nx = int(len(metadata["x"]))
    nt = int(len(metadata["t"]))

    arrays = {
        0: np.asarray(metadata["u"], dtype=float).reshape(nx, nt),
        1: np.asarray(metadata["u_x"], dtype=float).reshape(nx, nt),
        2: np.asarray(metadata["u_xx"], dtype=float).reshape(nx, nt),
        3: np.asarray(metadata["u_xxx"], dtype=float).reshape(nx, nt),
    }

    values = np.ones((nx, nt), dtype=float)
    for item in module_key:
        values *= arrays[int(item)]

    diff_x = int(diff_key[1])
    if diff_x > 0:
        values = finite_diff_x_array(values, x_values, diff_x)

    return values.reshape(nx * nt, 1)


def load_metadata_file(metadata_path):
    payload = np.load(metadata_path, allow_pickle=True)
    return {
        "x": np.asarray(payload["x"], dtype=float),
        "t": np.asarray(payload["t"], dtype=float),
        "u": np.asarray(payload["u"], dtype=float),
        "u_t": np.asarray(payload["u_t"], dtype=float),
        "u_tt": np.asarray(payload["u_tt"], dtype=float),
        "u_x": np.asarray(payload["u_x"], dtype=float),
        "u_xx": np.asarray(payload["u_xx"], dtype=float),
        "u_xxx": np.asarray(payload["u_xxx"], dtype=float),
    }


def regress_case_with_equation_form(metadata, equation_key, epi, delete_num):
    nx = int(len(metadata["x"]))
    nt = int(len(metadata["t"]))
    target = delete_boundary_flat(metadata["u_t"], nx, nt, delete_num)

    theta_columns = []
    for term_key in equation_key:
        term_values = metadata_term_array(metadata, term_key)
        theta_columns.append(delete_boundary_flat(term_values, nx, nt, delete_num))
    theta = np.hstack(theta_columns)

    coef, _, _, _ = np.linalg.lstsq(theta, target, rcond=None)
    residual = target - np.dot(theta, coef)
    mse = float(np.mean(residual ** 2))
    length_penalty_coef = sum(len(term_key[0]) for term_key in equation_key)
    fitness = mse + float(epi) * length_penalty_coef
    return coef.reshape(-1), mse, float(fitness)


def build_dimensionless_coefficient_dataset(
    results_path=None,
    out_dir=None,
    epi=2e-5,
    delete_num=10,
    frequency_summary=None,
):
    """
    Use the most frequent discovered equation form as the fixed structure.

    For every case, regress coefficients for that fixed structure, combine the
    coefficients with case variables, and save the dataset under
    data/dimensionless_dataset.
    """
    if results_path is None:
        results_path = os.path.join(RESULT_DIR, add_noise_suffix("all_discovered_equations.txt", NOISE_TAG))
    if out_dir is None:
        out_dir = DIMENSIONLESS_DATASET_DIR

    if frequency_summary is None:
        summary = summarize_equation_form_frequencies(results_path)
    else:
        summary = frequency_summary
    equation_key = summary["best_key"]
    term_labels = [term_to_text(list(module_key), list(diff_key)) for module_key, diff_key in equation_key]
    blocks = read_discovery_result_blocks(results_path)

    records = []
    coefficient_rows = []
    variable_names = ["U", "L", "D", "k", "Pe", "Da"]
    for block in blocks:
        case_name = block.get("case", "unknown_case")
        # Resolve metadata by case inside this package, including when the
        # result text was copied from a different project or machine.
        metadata_path = os.path.join(METADATA_DIR, "{}_ann_metadata.npz".format(case_name))
        if not os.path.exists(metadata_path):
            print("Skip {} because metadata file is missing: {}".format(case_name, metadata_path))
            continue

        metadata = load_metadata_file(metadata_path)
        variables = block.get("variables", {})
        coef, mse, fitness = regress_case_with_equation_form(metadata, equation_key, epi, delete_num)
        coefficient_rows.append(coef)

        record = {
            "case": case_name,
            "metadata": metadata_path,
            "mse": mse,
            "fitness": fitness,
        }
        for name in variable_names:
            record[name] = variables.get(name, np.nan)
        for index, value in enumerate(coef):
            record["coef_{}".format(index)] = float(value)
        records.append(record)

    if not records:
        raise ValueError("No valid case records were available for coefficient dataset generation.")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, add_noise_suffix("dimensionless_coefficients.csv", NOISE_TAG))
    npz_path = os.path.join(out_dir, add_noise_suffix("dimensionless_dataset.npz", NOISE_TAG))
    equation_path = os.path.join(out_dir, add_noise_suffix("equation_form.txt", NOISE_TAG))

    fieldnames = ["case", "metadata"] + variable_names + [
        "coef_{}".format(index) for index in range(len(term_labels))
    ] + ["mse", "fitness"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    variable_matrix = np.asarray(
        [[float(record[name]) for name in variable_names] for record in records],
        dtype=float,
    )
    coefficient_matrix = np.asarray(coefficient_rows, dtype=float)
    case_names = np.asarray([record["case"] for record in records], dtype=object)
    np.savez_compressed(
        npz_path,
        cases=case_names,
        variable_names=np.asarray(variable_names, dtype=object),
        variables=variable_matrix,
        term_labels=np.asarray(term_labels, dtype=object),
        coefficients=coefficient_matrix,
        mse=np.asarray([record["mse"] for record in records], dtype=float),
        fitness=np.asarray([record["fitness"] for record in records], dtype=float),
        equation_form=canonical_equation_text(equation_key),
    )

    with open(equation_path, "w", encoding="utf-8") as f:
        f.write("source: {}\n".format(results_path))
        f.write("selected_equation_form: {}\n".format(canonical_equation_text(equation_key)))
        f.write("frequency: {}/{}\n".format(summary["best_frequency"], summary["total_cases"]))
        f.write("term_labels: {}\n".format(term_labels))
        f.write("csv: {}\n".format(csv_path))
        f.write("npz: {}\n".format(npz_path))

    print("\nDimensionless coefficient dataset saved:")
    print(csv_path)
    print(npz_path)
    print(equation_path)

    return {
        "csv_path": csv_path,
        "npz_path": npz_path,
        "equation_path": equation_path,
        "equation_form": canonical_equation_text(equation_key),
        "term_labels": term_labels,
        "records": records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Discover PDE from a trained ANN surrogate model.")
    parser.add_argument("--case", default="case_000", help="Case name, for example case_001.")
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=525)
    parser.add_argument("--epi", type=float, default=2e-5)
    parser.add_argument("--delete_num", type=int, default=10)
    parser.add_argument("--criterion", default="standard", choices=["standard", "Galieo"])
    parser.add_argument("--pop_size", type=int, default=400)
    parser.add_argument("--n_generations", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=5)
    parser.add_argument("--run_all", action="store_true", default=True, help="Run GGA for every trained case model.")
    parser.add_argument("--single_case", action="store_false", dest="run_all", help="Run only --case.")
    parser.add_argument("--all_results_file", default="all_discovered_equations.txt")
    parser.add_argument("--build_dimensionless_dataset", action="store_true", default=True)
    parser.add_argument(
        "--no_build_dimensionless_dataset",
        action="store_false",
        dest="build_dimensionless_dataset",
    )
    parser.add_argument(
        "--only_build_dimensionless_dataset",
        action="store_true",
        help="Only read discovery_results and build data/dimensionless_dataset; do not run GGA.",
    )
    parser.add_argument(
        "--run_gga",
        action="store_true",
        default=True,
        help="Run the GGA discovery loop. Default remains dataset-only, matching the previous entry point.",
    )
    add_noise_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_noise_paths(args.noise_level)
    set_seed(args.seed)
    print("noise_level={} tag={}".format(NOISE_LEVEL, NOISE_TAG or "clean"))
    print("Case data directory: {}".format(CASE_DATA_DIR))
    print("Model directory: {}".format(MODEL_DIR))
    print("ANN metadata directory: {}".format(METADATA_DIR))
    print("Discovery results directory: {}".format(RESULT_DIR))
    print("GGA temporary directory: {}".format(TEMP_SAVE_DIR))
    print("Dimensionless dataset directory: {}".format(DIMENSIONLESS_DATASET_DIR))
    if NOISE_TAG and args.case == "case_000":
        args.case = add_noise_suffix("case_000", NOISE_TAG)

    if args.only_build_dimensionless_dataset:
        all_result_path = os.path.join(RESULT_DIR, add_noise_suffix(args.all_results_file, NOISE_TAG))
        dataset_result = build_dimensionless_coefficient_dataset(
            all_result_path,
            epi=args.epi,
            delete_num=args.delete_num,
        )
        print("\nDimensionless dataset finished.")
        print("CSV saved: {}".format(dataset_result["csv_path"]))
        print("NPZ saved: {}".format(dataset_result["npz_path"]))
        print("Equation form: {}".format(dataset_result["equation_form"]))
        return

    device = resolve_device(args.device)

    if args.run_all:
        all_result_path = run_all(args, device)
        summary = summarize_equation_form_frequencies(all_result_path)
        if args.build_dimensionless_dataset:
            dataset_result = build_dimensionless_coefficient_dataset(
                all_result_path,
                epi=args.epi,
                delete_num=args.delete_num,
                frequency_summary=summary,
            )
        print("\nAll cases finished.")
        print("All results saved: {}".format(all_result_path))
        print("Frequency summary saved: {}".format(summary["summary_path"]))
        if args.build_dimensionless_dataset:
            print("Dimensionless dataset saved: {}".format(dataset_result["csv_path"]))
        return

    case_name = normalize_case_name(args.case)
    result, metadata_path, variables, correct_result = discover_one_case(
        case_name,
        args,
        device,
        run_correct_check=True,
    )
    print("\nCorrect-library regression:")
    if correct_result.get("true_equation_text"):
        print("True equation:")
        print(correct_result["true_equation_text"])
    print("Regressed equation:")
    print(correct_result["equation_text"])
    print("MSE: {:.10e}".format(correct_result["mse"]))
    print("Fitness: {:.10e}".format(correct_result["fitness"]))

    result_path = save_result(case_name, result, metadata_path, variables, correct_result=correct_result)

    print("\nDiscovered equation:")
    print(result["equation_text"])
    print("Fitness: {:.10e}".format(result["best_fitness"]))
    print("Result saved: {}".format(result_path))


if __name__ == "__main__":
    args = parse_args()
    configure_noise_paths(args.noise_level)
    if args.run_gga:
        main()
    else:
        set_seed(args.seed)
        print("noise_level={} tag={}".format(NOISE_LEVEL, NOISE_TAG or "clean"))
        dataset_result = build_dimensionless_coefficient_dataset(
            epi=args.epi,
            delete_num=args.delete_num,
        )
        print("\nDimensionless dataset finished.")
        print("CSV saved: {}".format(dataset_result["csv_path"]))
        print("NPZ saved: {}".format(dataset_result["npz_path"]))
        print("Equation form: {}".format(dataset_result["equation_form"]))
