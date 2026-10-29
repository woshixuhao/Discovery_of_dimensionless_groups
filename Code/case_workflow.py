"""Load example cases, prepare their prior models, and support group inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from knowledge_aware_learning import (
    build_vocabs,
    filter_trainable_samples,
    load_prior_knowledge_database,
    print_specific_subset_debug,
    print_subset_candidates,
    train_model,
)


def normalize_case_key(value):
    key = str(value or "").strip().lower()
    key = key.replace("-", "_").replace(" ", "_")
    if key.startswith("example_"):
        key = key[len("example_") :]
    return key


def load_example_samples(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Cannot find example sample file: {}".format(path))

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def find_example_sample(samples, case_name):
    target = normalize_case_key(case_name)
    exact_matches = []
    fuzzy_matches = []

    for sample in samples:
        sample_id = sample.get("id", "")
        sample_name = sample.get("name", "")
        id_key = normalize_case_key(sample_id)
        name_key = normalize_case_key(sample_name)

        if target in {id_key, normalize_case_key("example_" + id_key)}:
            exact_matches.append(sample)
        elif id_key.endswith("_" + target) or id_key == target:
            exact_matches.append(sample)
        elif target and (target in id_key or target in name_key):
            fuzzy_matches.append(sample)

    matches = exact_matches or fuzzy_matches
    if not matches:
        known = ", ".join(sorted(normalize_case_key(s.get("id", "")) for s in samples))
        raise ValueError("Cannot find case '{}'. Known cases: {}".format(case_name, known))
    if len(matches) > 1:
        names = ", ".join(s.get("id", "") for s in matches)
        raise ValueError("Case name '{}' is ambiguous. Matches: {}".format(case_name, names))
    return matches[0]


def find_processed_case_dir(data_dir, case_name):
    data_root = Path(data_dir)
    target = normalize_case_key(case_name)
    direct = data_root / target
    if direct.exists() and direct.is_dir():
        return direct

    if data_root.exists():
        for path in data_root.iterdir():
            if path.is_dir() and normalize_case_key(path.name) == target:
                return path
    return None


def load_processed_case(data_dir, case_name):
    case_dir = find_processed_case_dir(data_dir, case_name)
    if case_dir is None:
        return None, None

    sample_path = case_dir / "sample.json"
    data_path = case_dir / "data.npz"
    metadata_path = case_dir / "metadata.json"

    if not sample_path.exists() or not data_path.exists():
        missing = []
        if not sample_path.exists():
            missing.append(str(sample_path))
        if not data_path.exists():
            missing.append(str(data_path))
        raise FileNotFoundError("Processed case is incomplete. Missing: {}".format(", ".join(missing)))

    with sample_path.open("r", encoding="utf-8") as f:
        sample = json.load(f)

    data = np.load(data_path)
    metadata = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    case_data = {
        "X": data["X"],
        "Y": data["Y"],
        "D_in": data["D_in"] if "D_in" in data else None,
        "metadata": metadata,
        "source": str(case_dir),
    }
    return sample, case_data


def data_file_candidates(data_dir, case_name, sample):
    case_key = normalize_case_key(case_name)
    sample_key = normalize_case_key(sample.get("id", ""))
    names = []
    for key in [case_key, sample_key]:
        if key and key not in names:
            names.append(key)
    paths = []
    for name in names:
        paths.append(Path(data_dir) / name / "data.npz")
        paths.append(Path(data_dir) / (name + ".npz"))
    return paths


def load_npz_case_data(data_dir, case_name, sample):
    for path in data_file_candidates(data_dir, case_name, sample):
        if path.exists():
            data = np.load(path)
            return {
                "X": data["X"],
                "Y": data["Y"],
                "D_in": data["D_in"] if "D_in" in data else None,
                "source": str(path),
            }
    return None


def reader_case_map():
    import process_example_datasets as examples

    mapping = {}
    for reader in examples.READERS:
        name = reader.__name__
        if name.startswith("read_"):
            mapping[name[len("read_") :]] = reader
    return mapping


def generate_case_data(case_name, examples_root):
    case_key = normalize_case_key(case_name)
    mapping = reader_case_map()
    if case_key not in mapping:
        known = ", ".join(sorted(mapping))
        raise ValueError(
            "No dataset reader for case '{}'. Reader cases: {}".format(case_name, known)
        )
    return mapping[case_key](Path(examples_root))


def save_case_data(record, data_dir):
    case_dir = Path(data_dir) / record["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "data.npz"
    np.savez_compressed(
        out_path,
        X=record["X"],
        Y=record["Y"],
        D_in=record["D_in"],
    )
    return out_path


def load_case_data(case_name, sample, data_dir, examples_root, cache_generated=True):
    data = load_npz_case_data(data_dir, case_name, sample)
    if data is not None:
        return data

    record = generate_case_data(case_name, examples_root)
    source = "generated by process_example_datasets.{}".format(
        record.get("data_source", "")
    )
    if cache_generated:
        path = save_case_data(record, data_dir)
        source = "{}; cached to {}".format(source, path)

    return {
        "X": record["X"],
        "Y": record["Y"],
        "D_in": record["D_in"],
        "source": source,
    }


def print_case_overview(sample, case_data):
    symbols = [var.get("symbol", "") for var in sample.get("variables", [])]
    metadata = case_data.get("metadata") or {}
    output_symbols = [
        col.get("symbol", "")
        for col in metadata.get("output_columns", [])
    ]

    print("\n" + "=" * 80)
    print("Example case")
    print("=" * 80)
    print("id              = {}".format(sample.get("id", "")))
    print("name            = {}".format(sample.get("name", "")))
    print("variables       = {}".format(symbols))
    if output_symbols:
        print("outputs         = {}".format(output_symbols))
    print("X shape         = {}".format(tuple(case_data["X"].shape)))
    print("Y shape         = {}".format(tuple(case_data["Y"].shape)))
    if case_data.get("D_in") is not None:
        print("D_in shape      = {}".format(tuple(case_data["D_in"].shape)))
    print("data source     = {}".format(case_data.get("source", "")))


def load_prediction_model(args, sample):
    case_key = normalize_case_key(args.case)
    case_specific_models = {"benard", "colebrook", "rayleigh", "mhd", "blasius"}
    selected_training_case = case_key if case_key in case_specific_models else None
    prior_samples = load_prior_knowledge_database(args.prior_dir, selected_training_case)
    train_samples, skipped = filter_trainable_samples(prior_samples)

    if case_key in case_specific_models:
        checkpoint_dir = Path(args.checkpoint_dir) / "case_specific" / case_key
        checkpoint_name = "knowledge_aware_learning_{}_best.pt".format(case_key)
        training_context = {
            "case_name": case_key,
            "n_train_samples": len(train_samples),
        }
    else:
        checkpoint_dir = args.checkpoint_dir
        checkpoint_name = "knowledge_aware_learning_best.pt"
        training_context = None

    if not train_samples:
        raise RuntimeError(
            "No trainable samples found in {}. Skipped {} samples.".format(
                args.prior_dir, len(skipped)
            )
        )

    vocabs = build_vocabs(train_samples + [sample])
    model = train_model(
        train_samples,
        vocabs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        force_retrain=args.force_retrain,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        training_context=training_context,
    )
    print("[Model] Using model: {}".format(Path(checkpoint_dir) / checkpoint_name))
    return model, vocabs


def parse_symbol_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(",") if item.strip()]


def debug_subset_from_symbols(model, sample, vocabs, symbols, args):
    symbols = parse_symbol_list(symbols)
    if not symbols:
        return []

    return print_specific_subset_debug(
        model,
        sample,
        vocabs,
        symbols,
        top_k_per_variable=args.debug_subset_top_k_per_variable,
        top_m_candidates=args.debug_subset_top_m_candidates,
        subset_size_penalty=args.subset_size_penalty,
        title="Debug specific subset on example case: {}".format(args.case),
    )


def parse_args():
    settings_parser = argparse.ArgumentParser(add_help=False)
    settings_parser.add_argument("--settings", default="settings.json")
    settings_args, remaining = settings_parser.parse_known_args()

    settings = {}
    settings_path = Path(settings_args.settings)
    if settings_path.exists():
        with settings_path.open("r", encoding="utf-8") as f:
            settings = json.load(f)

    parser = argparse.ArgumentParser(
        description="Predict dimensionless groups for one example dataset case."
    )
    parser.add_argument(
        "--settings",
        default=settings_args.settings,
        help="JSON settings file used as argument defaults.",
    )
    parser.add_argument(
        "case",
        nargs="?",
        default=settings.get("case", "benard"),
        help='Case name, for example "benard", "colebrook", or "wall_model".',
    )
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
    parser.add_argument("--top-n", type=int, default=settings.get("top_n", 5))
    parser.add_argument(
        "--subset-size-penalty",
        type=float,
        default=settings.get("subset_size_penalty", 1.0),
    )
    parser.add_argument(
        "--debug-subset-symbols",
        default=settings.get("debug_subset_symbols", []),
        help='Subset symbols to inspect, e.g. "h,delta_T,g,alpha,nu,kappa".',
    )
    parser.add_argument(
        "--debug-subset-top-k-per-variable",
        type=int,
        default=settings.get("debug_subset_top_k_per_variable", 5),
    )
    parser.add_argument(
        "--debug-subset-top-m-candidates",
        type=int,
        default=settings.get("debug_subset_top_m_candidates", 5),
    )
    return parser.parse_args(remaining)


def main():
    args = parse_args()
    print("[Case] Requested prediction case: {}".format(args.case))

    sample, case_data = load_processed_case(args.data_dir, args.case)
    if sample is None:
        samples = load_example_samples(args.samples_jsonl)
        sample = find_example_sample(samples, args.case)
        case_data = load_case_data(
            args.case,
            sample,
            args.data_dir,
            args.examples_root,
            cache_generated=not args.no_cache_generated_data,
        )
    print_case_overview(sample, case_data)

    model, vocabs = load_prediction_model(args, sample)
    debug_subset_from_symbols(
        model,
        sample,
        vocabs,
        ["u","c_p","T"],
        args,
    )


if __name__ == "__main__":
    main()
