# Reaction–Advection–Diffusion: PDE and Dimensionless-Number Discovery

This folder contains a workflow for generating reaction–advection–diffusion (RAD) data, training neural-network surrogate models, discovering PDE structures, and identifying dimensionless combinations that explain the discovered PDE coefficients. The scripts use modules and model weights within this folder; the original parent project is not required.

The synthetic data follow

$$
\frac{\partial c}{\partial t}
=-\frac{\partial c}{\partial x}
+\frac{1}{Pe}\frac{\partial^2 c}{\partial x^2}
+Da\,c-Da\,c^2,
\qquad Pe=\frac{UL}{D},\quad Da=\frac{kL}{U}.
$$

Here, `U` is velocity, `L` is characteristic length, `D` is diffusivity, and `k` is reaction rate. The field coordinates and PDE above are dimensionless. The generated reference coefficients are therefore `-1`, `1/Pe`, `Da`, and `-Da` for `c_x`, `c_xx`, `c`, and `c²`, respectively. The discovery code uses `u` to denote the same concentration field `c`.

## 1. Environment

Use Python 3.12 with the following packages:

```bash
python -m pip install numpy pandas torch matplotlib tqdm
```

The ANN training and PDE discovery commands default to CUDA. Use `--device cpu` if running on a CPU. Install a PyTorch build compatible with your CUDA environment when using a GPU. `tqdm` provides progress bars.

All commands below assume that the terminal is inside this folder:

```bash
cd Uploaded_RAD_code
```

## 2. Programs and Core Functions

| File | Purpose and main functions |
|---|---|
| [generate_data.py](./generate_data.py) | Generates synthetic RAD fields. `generate_parameter_cases()` enumerates physical parameters and calculates `Pe` and `Da`; `solve_one_case()` solves the PDE with periodic spatial differences and RK4 time integration; `generate_full_dataset()` generates or reuses clean fields and adds optional observation noise; `save_case_datasets()` exports individual cases; `generate_discovery_samples()` exports sampled field values and derivatives. |
| [train_surrogate_model.py](./train_surrogate_model.py) | Trains one ANN per case to approximate `(x, t) → c`. `NN` defines the network; `train_one_case()` trains and saves the checkpoint with the lowest validation loss. The current training activation is sine, with four hidden layers of 64 neurons by default. |
| [PDE_discovery.py](./PDE_discovery.py) | Loads the trained field models and searches for governing equations. `predict_with_derivatives()` computes the field and derivatives by automatic differentiation; `discover_one_case()` handles one case; `run_gga()` launches equation search; `run_all()` processes all matching models; `summarize_equation_form_frequencies()` counts discovered structures; `build_dimensionless_coefficient_dataset()` refits a shared structure across cases and exports the coefficient table. |
| [GGA.py](./GGA.py) | Implements the genetic equation search. `evolution()` initializes and evolves candidate PDEs; `translate_DNA()` evaluates candidate terms; `get_fitness()` estimates coefficients by SVD and scores residual error plus a complexity penalty; `cross_over()`, `mutation()`, and `select()` update the population. Called by the PDE discovery script. |
| [parameter_optimize_RAD.py](./parameter_optimize_RAD.py) | Discovers dimensionless dependence of varying PDE coefficients. `analyze_rad_coefficient_cv()` separates approximately constant and varying coefficients; `rad_discover_candidates_with_knowledge_model()` proposes dimensionally valid combinations; `choose_independent_base_pi()` selects independent candidates; `search_one_pi()` optimizes their exponents and regression degree; `export_best_rank_equation()` writes the final selected coefficient relationship. |
| [rad_knowledge.py](./rad_knowledge.py) | Defines the dimension-aware model and inference utilities. `load_checkpoint_model()` loads the supplied knowledge checkpoint, including its dimension and exponent metadata; `predict_distributions()` predicts exponent probabilities; `alpha_to_formula()` formats exponent vectors as formulas. This module does not retrain the supplied knowledge model. |
| [rad_optimization.py](./rad_optimization.py) | Supplies exponent enumeration, combined dimensionless-number construction, regression, complexity scoring, and logging. `compute_log_pi_matrix()` evaluates candidate groups in log space; `random_regression_split()` creates the reproducible data split; `fit_poly_1d_train_test()` fits on training data and evaluates held-out data. |
| [rad_noise.py](./rad_noise.py) | Parses noise levels and creates consistent case, checkpoint, and result filename suffixes. |
| [settings.json](./settings.json) | Configures dimensionless-number discovery, including input/output paths, the knowledge checkpoint, candidate counts, exponent options, regression degree, and scoring thresholds. |
| [knowledge_aware_learning_rad_best.pt](./knowledge_aware_learning_rad_best.pt) | Pretrained knowledge model used to propose dimensionless groups. This is separate from the per-case concentration-field models in `model/`. |

## 3. Data Formats

### Field data

The current parameter grid produces 78 cases after filtering by the allowed `Pe` and `Da` ranges. The default spatial and saved-time grids contain 128 and 101 points, respectively.

Each `case_XXX.npz` contains:

| Key | Meaning |
|---|---|
| `x`, `t` | Spatial and temporal coordinates, with lengths 128 and 101 under the default settings. |
| `C_clean` | Noise-free concentration field, shaped `(101, 128)` in time–space order. |
| `C_observed` | Observed field of the same shape; identical to the clean field at zero noise. |
| `U`, `L`, `D`, `k`, `Pe`, `Da` | Scalar physical parameters and reference dimensionless numbers. |
| `coef_cx_true`, `coef_cxx_true`, `coef_c_true`, `coef_c2_true` | Reference PDE coefficients from the generator. |
| `noise_level`, `noise_tag` | Observation-noise settings. |

The corresponding `case_XXX_variables.json` provides readable case metadata. The generator also writes `params.csv`, the aggregate `full_field_dataset.npz`, and, unless disabled, `discovery_samples.csv` with 1,000 sampled points per case. The sampled `c_t` is calculated from the known generating PDE; the ANN/GGA discovery stage instead computes derivatives from the trained surrogate.

The supplied `cases/` directory contains individual cases, while `rad_dataset/` contains bundled aggregate data. Fresh generation writes to `data/`. These aggregate files are not substitutes for the PDE coefficient table required by the final optimization stage.

### PDE coefficient data

The main input to dimensionless-number discovery is:

```text
data/dimensionless_dataset/dimensionless_coefficients.csv
```

Each row represents one case. Columns include `case`, `metadata`, `U`, `L`, `D`, `k`, `Pe`, `Da`, the fitted `coef_0`, `coef_1`, … values, and the regression `mse` and `fitness`.

**Coefficient indices follow the selected equation structure.** Read `term_labels` in `dimensionless_dataset.npz` or `equation_form.txt` to identify the PDE term associated with each column. Do not assume that a particular coefficient index always represents the same term.

The NPZ file also stores the case names, parameter matrix, coefficient matrix, and selected equation structure. Coefficients are refitted by least squares against `u_t` using the most frequent discovered right-hand-side structure; they are not simply copied from each case's original GGA result.

## 4. Running the Workflow

Run the stages in the following order. Existing outputs allow you to start at a later stage when its inputs are available.

### Step 1 — Generate field data

```bash
python generate_data.py --noise-level 0
```

This writes clean fields, per-case datasets, sampled data, and heatmaps under `data/`. To omit the optional heatmaps and sampling table:

```bash
python generate_data.py --noise-level 0 --no-heatmaps --no-discovery-samples
```

An existing `data/full_field_dataset.npz` can be reused. Add `--no-reuse-clean` to solve the fields again. To use the supplied per-case data directly, skip generation and pass `--data_dir cases` to the training command below.

### Step 2 — Train per-case field models

For newly generated data:

```bash
python train_surrogate_model.py --data_dir data/cases --noise-level 0 --device cuda
```

For the supplied individual cases:

```bash
python train_surrogate_model.py --data_dir cases --noise-level 0 --device cuda
```

Training defaults to 50,000 epochs per case, a learning rate of `0.001`, and a 20% validation fraction. Adjust these with `--epochs`, `--lr`, and `--val_fraction`. Checkpoints and training summaries are written to `model/`.

If the required `model/case_XXX_best.pt` checkpoints are already supplied, skip this stage.

### Step 3 — Discover PDEs and export the coefficient table

```bash
python PDE_discovery.py --noise-level 0 --device cuda
```

The default command processes all matching checkpoints, saves ANN derivative metadata, runs GGA, summarizes equation frequencies, and builds the coefficient dataset. Default search settings are a `100 × 100` discovery grid, 400 individuals, 200 generations, and `epi=2e-5`.

The current derivative grid is fixed to `x,t ∈ [0.1, 0.9]`. Case data are read from `data/cases/` if that directory exists, otherwise from `cases/`. Ensure the chosen directory contains the cases corresponding to the checkpoints.

To inspect a single case:

```bash
python PDE_discovery.py --single_case --case case_000 --noise-level 0 --device cuda
```

Single-case mode writes that case's equation and metadata; use the all-case workflow to construct the cross-case coefficient dataset.

If equation results and derivative metadata already exist, export the table without rerunning GGA:

```bash
python PDE_discovery.py --only_build_dimensionless_dataset --noise-level 0
```

This requires `discovery_results/all_discovered_equations.txt` and the matching `data/discovery_metadata/case_XXX_ann_metadata.npz` files. Metadata stored directly under `discovery_metadata/` must be placed in `data/discovery_metadata/` for the current path configuration. Cases with missing metadata are skipped.

### Step 4 — Discover dimensionless coefficient relationships

```bash
python parameter_optimize_RAD.py --noise-level 0
```

This stage reads `settings.json`, the coefficient dataset, and the local knowledge checkpoint. It does not train the concentration-field models or the knowledge model.

For each coefficient with `std/abs(mean) > cv_threshold`, it searches for a single combined dimensionless number formed from independent candidate groups. It fits `log(abs(coef))` as a polynomial in `log(Pi)`, recording coefficient signs separately. Search candidates use an 80% training / 20% held-out split with seed 42. Ranking balances held-out R² against complexity; the default maximum polynomial degree is 3. The configured R² threshold is a target, not a guarantee that every coefficient reaches it.

Useful overrides include:

```bash
python parameter_optimize_RAD.py --device cpu --data-dir data/dimensionless_dataset --out-dir discovery_results --checkpoint knowledge_aware_learning_rad_best.pt
```

Relative paths in this command and the local settings are resolved against this script's directory.

### Step 5 — Re-export final equations, if needed

Normal optimization already exports equation TXT files. To regenerate them from saved search PKL and NPZ files without rerunning the search:

```bash
python parameter_optimize_RAD.py --export-equations-only --noise-level 0
```

## 5. Output Locations and Interpretation

All paths below are relative to this folder.

| Location | Contents |
|---|---|
| `data/params.csv`, `data/full_field_dataset.npz` | Generated case parameters and aggregate concentration fields. |
| `data/cases/` | Generated per-case NPZ files, JSON metadata, and case index. |
| `data/discovery_samples.csv` | Optional sampled data from the generator. |
| `data/heatmaps/` | Optional field heatmaps in PNG/PDF format. |
| `model/` | Per-case best-validation checkpoints and CSV/JSON training summaries. |
| `data/discovery_metadata/` | ANN predictions and derivatives used for PDE discovery and coefficient refitting. |
| `temp_save/` | GGA temporary best-genome files. |
| `discovery_results/all_discovered_equations.txt` | Per-case discovered PDE structures, fitted coefficients, and fitness scores. |
| `discovery_results/case_XXX_discovered_equation.txt` | Equation and reference-library check from single-case mode. |
| `discovery_results/equation_form_frequency_summary.txt` | Frequencies of discovered structures and the most frequent right-hand-side structure. |
| `data/dimensionless_dataset/` | Coefficient CSV/NPZ, equation/term mapping, and coefficient CV summary TXT/CSV. |
| `discovery_results/rad_dimensionless_dataset_coef_*.npz` | Physical inputs, coefficient targets, transformed targets, and case names used in each coefficient search. |
| `discovery_results/rad_dimensionless_discovery_coef_*.txt` | Candidate rankings, exponent combinations, train/test scores, and final selected dimensionless number. |
| `discovery_results/rad_dimensionless_discovery_coef_*.pkl` | Saved candidate information and final selection for later equation export. |
| `discovery_results/rank2_linear_plots/*_rank2_linear_equation.txt` | Final best-candidate equation, combined exponent vector, polynomial degree, and training/test R². The directory and filenames retain their historical names; the contents use the final best candidate, **not Rank 2**. The current exporter does not generate plots. |

The final TXT files define `Pi` through `combined_form` and express the coefficient relationship as `log(abs(coef)) = P(log(Pi))`. The signed equation is `coef = sign × exp(P(log(Pi)))`. For a degree-one polynomial, this is a power law in `Pi`. The exporter fits on the same training rows used in selection and reports held-out performance separately. If coefficient signs vary, the signed expression uses the majority sign and the TXT records that limitation.

The reference relationships are `coef(u) ≈ Da`, `coef(u²) ≈ -Da`, `coef(u_x) ≈ -1`, and `coef(u_xx) ≈ 1/Pe`. Equivalent reciprocal groups, such as `1/Da` or `Pe`, can represent the same dependence through a negative fitted exponent. Discovered coefficients and formulas approximate these reference relationships rather than being constrained to equal them exactly.

## 6. Optional Noisy Cases

Use the same explicit noise level throughout generation, training, PDE discovery, and dimensionless optimization. For example:

```bash
python generate_data.py --noise-level 0.05
python train_surrogate_model.py --data_dir data/cases --noise-level 0.05 --device cuda
python PDE_discovery.py --noise-level 0.05 --device cuda
python parameter_optimize_RAD.py --noise-level 0.05
```

Nonzero levels add suffixes such as `_5pct` to case, checkpoint, and result filenames. Zero noise uses unsuffixed names. Run any script with `--help` to inspect its supported options.
