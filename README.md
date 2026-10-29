# Discovery_of_dimensionless_variable

Data and code for discovering dimensionless variables through dimensional prior knowledge and data-driven optimization. A Set Transformer learns exponent distributions from known dimensionless quantities. Dimensional constraints then generate candidate groups, and numerical case data select useful products of these groups or a pair of groups for predicting a physical response.

# Environments

The following versions were inspected in the current development environment:

```text
python==3.12.7
numpy==1.26.4
torch==2.3.1
tqdm==4.66.4
matplotlib==3.10.8
```

NumPy and PyTorch are required for the main workflow. `tqdm` provides optional progress bars; the code falls back to text progress messages when it is unavailable. The exponent-search and polynomial-regression workflow uses NumPy for regression. Matplotlib is used for the figures in the Rayleigh and Rayleigh-Benard example notebooks.

```bash
python -m pip install numpy==1.26.4 torch==2.3.1 tqdm==4.66.4 matplotlib==3.10.8
```

PyTorch automatically selects CUDA when available, otherwise CPU. Use `--device cpu` or `--device cuda` to select a device explicitly. The supplied processed datasets can be loaded without the packages used to generate the original datasets or notebooks.

# The dataset utilized in this work

The release contains two different types of data: a **prior knowledge database for learning exponent distributions**, and **numerical example datasets for selecting predictive dimensionless groups**.

## 1. Dimensionless prior knowledge database

[prior_knowledge_database/](./prior_knowledge_database/) contains **1,310 JSON files** exported from a database of dimensionless physical quantities. The source collection was constructed from Josef Kuneš, *Dimensionless Physical Quantities in Science and Engineering* (Elsevier, 2012). Each file describes one multiplicative representation of a known dimensionless quantity, rather than a table of experimental observations.

For variables `x_1, ..., x_p`, a record represents

```text
Pi = x_1^alpha_1 * x_2^alpha_2 * ... * x_p^alpha_p.
```

The exponent `alpha_i` is the supervised label for variable `x_i`. A zero exponent excludes that variable; a negative exponent places it in the denominator. Fractional exponents can be stored as fraction strings such as `"1/2"`.

The fields have the following meanings:

| Field | Meaning |
| --- | --- |
| `source_id` | Identifier of the source quantity, such as `3.1.83` for the Reynolds number. |
| `name` | Name of the dimensionless quantity. |
| `variables` | Ordered list of variables in this representation. The exponent vector follows this order. |
| `variables[i].name` | Physical variable name. |
| `variables[i].unit` | Human-readable unit annotation. Numerical dimensional calculations use `dimension_vector`. |
| `variables[i].dimension_vector` | Powers of the seven SI base dimensions. Missing base-dimension keys are treated as zero. |
| `variables[i].alpha` | Exponent of this variable in the known dimensionless product. |
| `source` | Provenance, including section, quantity number, symbols, page and reference identifiers where available. These identifiers are source metadata, not links to bundled bibliography files. |

The seven base dimensions are encoded in the fixed order:

| Key | Base dimension |
| --- | --- |
| `M` | Mass |
| `L` | Length |
| `T` | Time |
| `I` | Electric current |
| `Theta` | Thermodynamic temperature |
| `N` | Amount of substance |
| `J` | Luminous intensity |

Here `J` denotes a base-dimension key, not the joule unit. For example, velocity has `L=1, T=-1`, and kinematic viscosity has `L=2, T=-1`; all other entries are zero.

An abbreviated version of the Reynolds-number record [3.1.83-1.json](./prior_knowledge_database/3.1.83-1.json) illustrates the format:

```json
{
  "source_id": "3.1.83",
  "name": "Reynolds Number",
  "variables": [
    {
      "name": "flow velocity",
      "unit": "m s^-1",
      "dimension_vector": {"M": 0, "L": 1, "T": -1, "I": 0, "Theta": 0, "N": 0, "J": 0},
      "alpha": 1
    },
    {
      "name": "characteristic length",
      "unit": "m",
      "dimension_vector": {"M": 0, "L": 1, "T": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
      "alpha": 1
    },
    {
      "name": "kinematic viscosity",
      "unit": "m^2 s^-1",
      "dimension_vector": {"M": 0, "L": 2, "T": -1, "I": 0, "Theta": 0, "N": 0, "J": 0},
      "alpha": -1
    }
  ]
}
```

This gives `Re = velocity * length / kinematic_viscosity`, with exponent vector `[1, 1, -1]`. Arrange the variable dimension vectors as columns of a matrix `D`; a valid dimensionless product satisfies `D @ alpha = 0`.

Files with suffixes such as `-1`, `-2` and `-3` may contain different representations of the same source quantity. Therefore, `source_id` is not necessarily unique across files. The source name or a full formula string is not needed as a neural input: the model receives only each variable's seven dimensional exponents. Names, units, domain labels and physical descriptions are retained for interpretation.

The loader normalizes records and expands the exponent-class set using the labels present in the database. Across the full database, the current dimensional consistency check accepts **1,288 records** and skips **22 records** whose stored exponents do not satisfy the dimensionless constraint. These records remain in the released database for traceability. Case-specific exclusions described below reduce the training set further.

## 2. Processed numerical example datasets

Each case in [example_processed_data/](./example_processed_data/) contains the same eight files:

| File | Contents and use |
| --- | --- |
| `data.npz` | Compressed arrays `X`, `Y` and `D_in`; this is the numerical file read by the main workflow. |
| `X.npy` | Input matrix, identical to `X` in the NPZ archive. |
| `Y.npy` | Response matrix, identical to `Y` in the NPZ archive. |
| `D_in.npy` | Input dimension matrix, identical to `D_in` in the NPZ archive. |
| `X.csv` | Human-readable input table with a header. |
| `Y.csv` | Human-readable response table with a header. |
| `sample.json` | Case description and ordered input variables, including their symbols and seven-dimensional vectors. These inference records do not require known `alpha` labels. |
| `metadata.json` | Array shapes, dimension-row order, input/output column descriptions, source information and filenames. |

For a case with `n` observations, `p` dimensional input variables and `q` response columns:

```text
X.shape    = (n, p)
Y.shape    = (n, q)
D_in.shape = (number_of_dimension_rows, p)
```

Row `i` of `X` and row `i` of `Y` describe the same observation. Column `j` of `X`, entry `j` of `sample.json["variables"]`, and column `j` of `D_in` refer to the same input variable. All supplied cases have one response column, so the default `--output-index 0` selects it.

**Read `metadata.json["row_dims"]` before interpreting `D_in`.** Its row order is case-specific and differs from the seven-dimensional neural input order. For example, Colebrook uses `['L', 'T', 'M']`, whereas Bénard uses `['L', 'T', 'M', 'Theta']`. The JSON dimension vectors explicitly name their base dimensions and remove this ambiguity. The metadata field `num_input` is inherited from the original example setup; it is not the number of columns in `X`. Use `X.shape[1]` or `input_columns` for the actual input-variable count.

Example: inspect the Colebrook data from the release directory.

```python
import json
from pathlib import Path
import numpy as np

case_dir = Path("example_processed_data/colebrook")
with np.load(case_dir / "data.npz") as data:
    X, Y, D_in = data["X"], data["Y"], data["D_in"]
sample = json.loads((case_dir / "sample.json").read_text(encoding="utf-8"))
metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))

print(X.shape, Y.shape, D_in.shape)  # (100000, 5), (100000, 1), (3, 5)
print([v["symbol"] for v in sample["variables"]])  # u, rho, D, k, mu
print(metadata["row_dims"])  # L, T, M

# Re = u * rho * D / mu, using the stored column order.
alpha_Re = np.array([1, 1, 1, 0, -1])
assert np.allclose(D_in @ alpha_Re, 0)
Re = np.exp(np.log(X) @ alpha_Re)
```

### Cases included in this release

The current checkout contains seven processed cases.

| Case | Observations | Input columns in stored order | Response in `Y` | `D_in` row order |
| --- | ---: | --- | --- | --- |
| [benard](./example_processed_data/benard/) | 182 | `h, delta_T, lambda, g, alpha, nu, kappa` | Nusselt number `Nu` | `L, T, M, Theta` |
| [blasius](./example_processed_data/blasius/) | 10,000 | `U, nu, x, y` | Streamwise velocity ratio `u_over_U` | `L, T` |
| [colebrook](./example_processed_data/colebrook/) | 100,000 | `u, rho, D, k, mu` | Stored friction-factor response `f` | `L, T, M` |
| [dixit](./example_processed_data/dixit/) | 144 | `U_inf, nu, M, delta, delta_star, theta` | Stored boundary-layer response `Y` | `L, T` |
| [keyhole](./example_processed_data/keyhole/) | 90 | `etaP, Vs, rho, cp, Tl-T0, r0, alpha` | Normalized keyhole depth `e_star` | `L, T, M, Theta` |
| [mhd](./example_processed_data/mhd/) | 483 | `l, mu, rho, eta, dp_dx, B` | Velocity Reynolds number `Re_u` | `L, T, M, I` |
| [rayleigh](./example_processed_data/rayleigh/) | 100,000 | `U, y, t, nu` | Velocity ratio `u_over_U` | `L, T` |

- **Bénard convection:** Data exported from `dataset_rb.csv`. Inputs describe layer height `h`, temperature difference `delta_T`, thermal conductivity `lambda`, gravitational acceleration `g`, thermal expansion coefficient `alpha`, kinematic viscosity `nu`, and thermal diffusivity `kappa`. The response is the Nusselt number, computed during preprocessing from the source heat-transfer quantities. Familiar reference groups are `Ra = g * alpha * delta_T * h^3 / (nu * kappa)` and `Pr = nu / kappa`. These illustrate physical interpretation; their recovery is not hard-coded into the optimizer.

- **Blasius boundary layer:** Synthetic data generated from a Blasius ODE solution on a 100-by-100 grid of streamwise and wall-normal positions. `U` is the free-stream velocity, `nu` is kinematic viscosity, and `x, y` are positions. The supplied dataset fixes `U=0.01` and `nu=1e-6`. The response is `u/U`; a familiar similarity coordinate is `y * sqrt(U / (nu * x))`. Zero-valued responses occur in this export, so its default target transformation is the raw response.

- **Colebrook pipe friction:** Synthetic samples generated by varying Reynolds number and relative roughness, with `rho=200`, `mu=0.001`, and pipe diameter `D` sampled between 1 and 10. Columns `u` and `k` are velocity and absolute roughness. The sampling ranges are approximately `10^3 <= Re <= 10^5` and `10^-8 <= k/D <= 10^-0.7`. Relevant dimensionless coordinates include `Re = rho*u*D/mu` and relative roughness `k/D`. The preprocessing iteration uses `f_new = 1 / (-log10((k/D)/3.7 + 5.02/(Re*sqrt(f))))^2`. This defines the exported response convention; do not substitute a differently normalized friction factor when comparing predictions with `Y`.

- **Dixit turbulent boundary layer:** Data exported from `Dixit_data.npz`. `U_inf` is free-stream velocity, `nu` is kinematic viscosity, `M` is the supplied integral parameter with dimension `L^3 T^-2`, and `delta`, `delta_star`, `theta` are boundary-layer, displacement and momentum thicknesses. In the stored data, `U_inf=1` and `delta=1`, reflecting the source normalization. The source notebook plots the response as `u_tau/U_inf`, but the processed export retains the generic output name `Y` and leaves input unit strings empty. Use the dimension vectors and existing normalization when reproducing this case. The original zero/adverse/favorable pressure-gradient index arrays are not included in this processed export.

- **Laser keyhole:** Data exported from `dataset_keyhole.csv`. Inputs are absorbed laser power `etaP`, scan speed `Vs`, density `rho`, specific heat `cp`, liquidus-to-initial temperature difference `Tl-T0`, beam radius `r0`, and thermal diffusivity `alpha`. The response `e_star` is the source column `e*`, representing normalized keyhole depth. The processed dimension matrix has already been reordered to match these seven input columns.

- **MHD generator:** Data exported from `MHD_Generator_Data_Uavg.csv`. Inputs are characteristic length `l` (fixed at 1), dynamic viscosity `mu`, density `rho`, electrical parameter `eta`, pressure-gradient quantity `dp_dx`, and magnetic field `B`. The response is `Re_u = u*rho*l/mu`, based on the source average velocity `u`. **Annotation inconsistency:** the stored unit string for `eta` is `S m^-1`, while its dimension vector is `M L^3 T^-3 I^-2`, which corresponds to electrical resistivity (`ohm m`). The model and dimensional constraints use the stored vector. This release preserves the data; the electrical convention needs to be resolved before interpreting results in physical units.

- **Rayleigh moving plate:** Synthetic velocity profiles for an impulsively moved plate, using `u/U = 1 - erf(y / (2*sqrt(nu*t)))`. The data combine 100 sampled `U, nu` pairs, 50 wall-normal positions and 20 times. `U` is plate speed, `y` is distance from the plate, `t` is time and `nu` is kinematic viscosity. A familiar similarity coordinate is `y/sqrt(nu*t)`. There are 2,000 rows at `y=0`; rows involving a nonzero exponent of this zero-valued input are excluded by logarithmic candidate evaluation. Zero responses also occur, so `auto_log` keeps the response in raw scale.

# How to reproduce our work?

## 1. Release files and completeness

Run commands from this directory so relative data, settings and output paths resolve correctly:

```bash
cd Upload_version
```

| File or directory | Purpose |
| --- | --- |
| [knowledge_aware_learning.py](./knowledge_aware_learning.py) | Load and validate prior records; encode dimensions; define and train the Set Transformer; predict exponents; decode dimensionally valid groups; enumerate variable subsets. |
| [case_workflow.py](./case_workflow.py) | Shared case workflow: load case data, normalize case names, select prior-training exclusions and checkpoints, and load/train the prediction model for dimensionless-group inference. |
| [parameter_optimize.py](./parameter_optimize.py) | Main reproduction entry point: generate candidate groups, evaluate them on numerical data, search their product exponents, fit response models, and save results. |
| [Rayleigh_example.ipynb](./Rayleigh_example.ipynb) | Interactive Rayleigh workflow with dimensional-data inspection, prior-model training, candidate generation, the released optimizer, response plots, physical validation and downloadable result exports. |
| [Rayleigh_benard_example.ipynb](./Rayleigh_benard_example.ipynb) | Interactive convection workflow covering all 182 observations, prior-model training, dimensionless-group discovery, Nusselt-response optimization, prediction plots and post-discovery interpretation using Rayleigh/Prandtl reference groups. |
| [settings.json](./settings.json) | Defaults for the selected case, training, candidate search, regression and output selection. Command-line arguments override these defaults. |
| [prior_knowledge_database/](./prior_knowledge_database/) | Supervised dimensional knowledge used to train the exponent classifier. |
| [example_processed_data/](./example_processed_data/) | The seven currently bundled, self-contained numerical examples described above. |

All seven currently bundled processed case directories contain their eight advertised files. The NPZ, NPY and CSV arrays agree, and the stored column descriptions and dimensional matrices agree. The three Python source files pass syntax parsing. A standalone CPU smoke run on the Bénard case completed one training epoch, candidate generation, optimization and result serialization. This verifies the workflow with reduced settings; it does not establish convergence or reproduce full-training scientific results. The Rayleigh notebook additionally contains the outputs of its complete 300-epoch demonstration run.

The following referenced resources are **not bundled**:

- `example_samples.jsonl`, `process_example_datasets.py`, and the original `Examples/` datasets/notebooks. These support the legacy fallback and raw-data regeneration path. They are unnecessary when using the complete processed case directories, which are loaded first. Source paths inside JSON files are provenance annotations.
- Pretrained `model_save/` checkpoints and previously computed `results_save/` outputs. These directories are created during a run. Exact reuse of an earlier trained model requires supplying its checkpoint.
- A standalone plotting script or saved paper figures. The command-line workflow generates text reports and serialized results; the Rayleigh notebook adds its own figures and embeds them in its cell outputs.

Two debug remnants affect the alternative standalone entry points in the current source: running [knowledge_aware_learning.py](./knowledge_aware_learning.py) directly reaches an unconditional `raise OSError` before training, and [case_workflow.py](./case_workflow.py) currently inspects the hard-coded subset `['u', 'c_p', 'T']`, which is absent from the bundled cases. Use [parameter_optimize.py](./parameter_optimize.py) for the full workflow, or the inference snippet below to call the reusable functions directly. These debug blocks do not execute when their modules are imported by the optimization entry point.

## 2. Train the model and optimize dimensionless groups

### Interactive Rayleigh-Benard example

Open [Rayleigh_benard_example.ipynb](./Rayleigh_benard_example.ipynb) and run all cells to discover dimensionless groups for convection heat transfer. This is the `benard` case, distinct from the impulsively moved plate in the Rayleigh notebook. Both notebooks use the same release functions and provide the same online repository-initialization options.

The default `demo` profile trains a fresh prior model for 300 epochs and optimizes using all 182 observations; `PROFILE = "full"` uses 3,000 epochs. The response is `Nu`, and the default preprocessing fits `log(Nu)`. The notebook reports metrics on both response scales, verifies the refitted model against the saved optimization score, and compares the discovered exponent vectors with the classical Rayleigh and Prandtl directions after discovery. A separately fitted Rayleigh power-law baseline is a diagnostic reference, not an exact analytical convection law or an input to the discovery algorithm.

Notebook outputs are isolated under `results_save/benard_notebook/<profile>/` and include the full response-model coefficients, predictions, split indices, figures, logs, a JSON summary and a ZIP archive. Saved cell outputs provide a GitHub preview; generated files are created in the runtime when the notebook executes.

### Command-line reproduction

After installing the packages, run a case:

```bash
python parameter_optimize.py colebrook
```

The current [settings.json](./settings.json) selects `colebrook` by default, with 3,000 training epochs, batch size 100,000, learning rate 0.001, 10 requested base candidates, maximum polynomial degree 5, and an R² threshold of 0.92. Thus `python parameter_optimize.py` also selects this case. Batch size applies to prior-knowledge records, not to the numerical case observations.

The same entry point runs every bundled example:

```bash
python parameter_optimize.py benard
python parameter_optimize.py blasius
python parameter_optimize.py colebrook
python parameter_optimize.py dixit
python parameter_optimize.py keyhole
python parameter_optimize.py mhd
python parameter_optimize.py rayleigh
```

These commands execute the full configured workflow. Runtime depends on hardware, training epochs, variable-subset count, candidate rank and exponent-search settings. CPU execution is supported; the neural stage can use a CUDA GPU.

For a short installation check, the following reduced run trains for one epoch and searches a much smaller space:

```bash
python parameter_optimize.py benard --device cpu --epochs 1 --batch-size 64 --top-k 3 --independent-rank-top-n 3 --rank-source-top-k 3 --classic-exponents=-1,0,1 --polynomial-max-degree 2 --search-refine-top-m 3 --force-retrain --force-retain
```

This command writes a low-epoch checkpoint to the normal Bénard checkpoint location. Use `--force-retrain --force-retain` again for a subsequent full run so the installation-check model and candidates are replaced.

### Core training and candidate-generation functions

In [knowledge_aware_learning.py](./knowledge_aware_learning.py):

| Function or class | Role |
| --- | --- |
| `load_prior_knowledge_database()` | Read JSON records, normalize their schema, exclude selected case-related prior quantities and build the available exponent classes. |
| `filter_trainable_samples()` | Retain records with valid labels and dimension vectors satisfying `D @ alpha = 0`. |
| `build_vocabs()` / `encode_variable()` | Define and encode the seven-dimensional input features. No name or domain embedding is used. |
| `DimensionlessDataset` / `collate_batch()` | Construct labeled variable sets and pad/mask sets of different sizes for minibatches. |
| `SetTransformerExponentClassifier` | Use attention across a variable set to output one exponent-class distribution per variable. The default network has hidden dimension 128, four attention heads and three attention blocks. |
| `train_model()` | Train using AdamW and cross-entropy loss, ignoring padded labels. Save the model with the lowest training loss, or load a compatible existing checkpoint. |
| `predict_distributions()` | Compute per-variable exponent probabilities for a new variable set. |
| `constrained_decode()` | Search likely exponent combinations and retain nonzero vectors satisfying the dimensional constraint. If that search finds none, use a Buckingham–Pi nullspace fallback. |
| `evaluate_subset_candidates()` | Enumerate subsets of input variables, rerun inference on each subset, and rank the valid groups using model log-probability, subset-size penalty and exponent-complexity penalty. This allows irrelevant variables to be omitted. |
| `alpha_to_formula()` / `print_subset_candidates()` | Convert exponent vectors to readable products and report ranked candidates. |

In [case_workflow.py](./case_workflow.py), `load_processed_case()` reads a case's description and numerical arrays. `load_prediction_model()` chooses the training database exclusions and checkpoint, then invokes `train_model()`.

For `benard`, `blasius`, `colebrook`, `mhd` and `rayleigh`, the model excludes named prior quantities associated with the case. The current exclusion and filtering counts are:

| Case | Prior records excluded by name | Training records retained after dimensional validation |
| --- | ---: | ---: |
| `benard` | 8 | 1,280 |
| `blasius` | 6 | 1,282 |
| `colebrook` | 8 | 1,280 |
| `mhd` | 7 | 1,281 |
| `rayleigh` | 8 | 1,280 |
| `dixit`, `keyhole` | 0 | 1,288 |

The global prior model applies to `dixit` and `keyhole`. These exclusions remove exact names listed in the loader; they do not guarantee removal of every algebraically related prior quantity.

### Core dimensionless-number optimization functions

The optimizer combines base groups using a second exponent vector `beta`:

```text
Pi_j       = product_i x_i^(alpha_ji)
Pi_total   = product_j Pi_j^(beta_j)
log(Pi_total) = sum_j beta_j * log(Pi_j)
gamma_i    = sum_j beta_j * alpha_ji
Pi_total   = product_i x_i^(gamma_i)
```

`alpha` therefore describes a base group, `beta` combines base groups, and `gamma` is the resulting exponent vector in the active physical-variable order. A dimensionless group remains dimensionless under these products and powers.

`run_single_target()` follows the candidate-rank and discrete exponent-search workflow above. Legacy variance/correlation, ElasticNet/Lasso and residual-diagnostic configuration options do not activate additional processing. Likewise, the `nn_*` and `beta_lr` settings do not activate a neural `beta` optimizer in this release; the released optimizer searches discrete exponents. The parsed `two_pi_exhaustive_search` setting does not expand the active four-feature two-group search.

## 3. Candidate inference without numerical optimization

Use the following Python snippet from the release directory to train/load the prior model and print candidate groups without invoking the standalone debug blocks:

```python
import json
from pathlib import Path
from types import SimpleNamespace
import case_workflow as workflow
from knowledge_aware_learning import set_seed, print_subset_candidates

set_seed(42)
args = SimpleNamespace(**json.loads(Path("settings.json").read_text(encoding="utf-8")))
args.case = "colebrook"
sample, case_data = workflow.load_processed_case(args.data_dir, args.case)
workflow.print_case_overview(sample, case_data)
model, vocabs = workflow.load_prediction_model(args, sample)
print_subset_candidates(
    model, sample, vocabs,
    top_n=args.top_n,
    subset_size_penalty=args.subset_size_penalty,
)
```

This stage uses dimensional descriptions to propose groups. The numerical response affects the later optimization stage, not the exponent-classifier training loss.

## 4. Checkpoints, settings and repeat runs

- Use `--settings path/to/config.json` for another configuration file. Use `python parameter_optimize.py --help` to inspect command-line options.
- Change `--target-transform` to `raw`, `log` or `auto_log`. Explicit `log` removes nonpositive responses. R² is evaluated on the chosen target scale.
- Use `--force-retrain` to retrain the exponent classifier and `--force-retain` to regenerate cached base candidates. After changing the prior database, model or dimensional annotations, use both flags: cached-candidate metadata does not fingerprint checkpoint weights or all source contents.
- The regression split has a fixed seed, but the default optimization CLI does not call `set_seed()` before neural training. For a seeded full run, use the command below. Hardware-dependent PyTorch behavior can still affect exact numerical results.

```bash
python -c "import knowledge_aware_learning as k; import parameter_optimize as p; k.set_seed(42); p.run_logged_main(p.parse_args())" colebrook --force-retrain --force-retain
```

The reported test split is reused to rank `beta`, select polynomial degrees and select group pairs. Its R² is a model-selection score, not performance on a separate untouched final evaluation set. For an independent generalization assessment, reserve additional observations outside the current selection workflow.

# Expected outputs

The model checkpoint and the case report are written separately:

```text
model_save/
    knowledge_aware_learning_best.pt                 # global prior model
    case_specific/<case>/
        knowledge_aware_learning_<case>_best.pt       # case-specific prior model

results_save/<case>/
    parameter_optimize_output.txt
    raw_candidates.pkl
    parameter_optimize_result.pkl
```

| Output | What it records |
| --- | --- |
| `*.pt` | Exponent-classifier weights, dimensional/exponent-class metadata, saved epoch and lowest training loss. This is the neural prior model, not a fitted response-regression model. |
| `parameter_optimize_output.txt` | Case and array information, training/loading messages, ranked candidate formulas, dimensional exponents, valid-row counts, target transformation, regression/search scores, final one- or two-group result and timing information. A subsequent run overwrites this log. Progress bars may appear only in the terminal. |
| `raw_candidates.pkl` | Dictionary containing candidate-cache `metadata` and `raw_candidates`, including subset information, formulas, full exponent vectors and prior-ranking scores. |
| `parameter_optimize_result.pkl` | Dictionary containing run `metadata`, deduplicated `candidates`, selected candidate indices `selected`, and final `result`. The final result records `kind` (`one` or `two`), selected step, metric key and a compact result row. |

Interpret the reported results as follows:

- **Candidate formula / `full_alpha`:** A base dimensionless product and its exponents in the active input order. Zero entries identify omitted variables. `D @ alpha = 0` confirms dimensional consistency, not predictive usefulness.
- **Model log-probability and candidate score:** Compatibility with the learned prior, adjusted by subset-size and exponent-complexity penalties. These scores are not R² values or probabilities that a physical law is correct. Some candidates may come from the Buckingham–Pi fallback.
- **`beta`, combined `alpha`, combined formula:** For a one-group result, these show how base groups combine and which powers of the physical inputs define `Pi_total`.
- **Powers and reciprocals:** Base-group canonicalization reduces proportional exponent vectors and fixes their overall sign. A familiar group may therefore appear as an equivalent power or reciprocal. Later `beta` optimization can choose these powers explicitly; their regression fits need not have identical scores.
- **Two-group result:** Two formulas and exponent vectors define the coordinates of the fitted response surface. The log distinguishes pair-selection R² from the final polynomial-in-log-groups R².
- **Train/test R² and method:** These describe the response fit on the reported target scale. If the target is `log(Y)`, converting predictions back to the original response requires exponentiation. The method indicates whether regression used `log(Pi)`, raw `Pi`, or two-group log-polynomial features. The threshold controls the search branch; a returned result does not guarantee that threshold was met.

To inspect the serialized result:

```python
import pickle

with open("results_save/colebrook/parameter_optimize_result.pkl", "rb") as f:
    payload = pickle.load(f)
print(payload["metadata"])
print(payload["result"])
```

The command-line compact result contains group definitions and fit metrics, but does not retain the full fitted polynomial coefficients, predictions or train/test row indices. Refit the response relation from the supplied arrays and recorded settings when these are needed. The Rayleigh notebook demonstrates this refit, verifies the reported R² and exports the complete polynomial and predictions. Figures are generated by the notebook rather than the command-line scripts. No pretrained checkpoints are bundled; the notebook's embedded outputs show its executed demonstration run.
