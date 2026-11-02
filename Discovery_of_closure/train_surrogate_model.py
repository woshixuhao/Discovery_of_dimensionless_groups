import argparse
import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rad_noise import add_noise_argument, add_noise_suffix, case_npz_regex, noise_tag, parse_noise_level


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "cases")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
ACTIVATION_FUNCTION = "Sin"



class Rational(torch.nn.Module):
    def __init__(self,
                 Data_Type = torch.float32,
                 Device    = torch.device('cpu')):
        # This activation function is based on the following paper:
        # Boulle, Nicolas, Yuji Nakatsukasa, and Alex Townsend. "Rational neural
        # networks." arXiv preprint arXiv:2004.01902 (2020).

        super(Rational, self).__init__()

        # Initialize numerator and denominator coefficients to the best
        # rational function approximation to ReLU. These coefficients are listed
        # in appendix A of the paper.
        self.a = torch.nn.parameter.Parameter(
                        torch.tensor((1.1915, 1.5957, 0.5, .0218),
                                     dtype = Data_Type,
                                     device = Device))
        self.a.requires_grad_(True)

        self.b = torch.nn.parameter.Parameter(
                        torch.tensor((2.3830, 0.0, 1.0),
                                     dtype = Data_Type,
                                     device = Device))
        self.b.requires_grad_(True)

    def forward(self, X : torch.tensor):
        """ This function applies a rational function to each element of X.
        ------------------------------------------------------------------------
        Arguments:
        X: A tensor. We apply the rational function to every element of X.
        ------------------------------------------------------------------------
        Returns:
        Let N(x) = sum_{i = 0}^{3} a_i x^i and D(x) = sum_{i = 0}^{2} b_i x^i.
        Let R = N/D (ignoring points where D(x) = 0). This function applies R
        to each element of X and returns the resulting tensor. """

        # Create aliases for self.a and self.b. This makes the code cleaner.
        a = self.a
        b = self.b

        # Evaluate the numerator and denominator. Because of how the * and +
        # operators work, this gets applied element-wise.
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X))
        D_X = b[0] + X*(b[1] + b[2]*X)

        # Return R = N_X/D_X. This is also applied element-wise.
        return N_X/D_X

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        x = torch.sin(x)
        return x

class NN(torch.nn.Module):
    def __init__(self,
                 Num_Hidden_Layers   : int          = 3,
                 Neurons_Per_Layer   : int          = 20,   # Neurons in each Hidden Layer
                 Input_Dim           : int          = 1,    # Dimension of the input
                 Output_Dim          : int          = 1,    # Dimension of the output
                 Data_Type           : torch.dtype  = torch.float32,
                 Device              : torch.device = torch.device('cpu'),
                 Activation_Function : str          = "Tanh",
                 Batch_Norm          : bool         = False):
        # For the code below to work, Num_Hidden_Layers, Neurons_Per_Layer,
        # Input_Dim, and Output_Dim must be positive integers.
        assert(Num_Hidden_Layers   > 0), "Num_Hidden_Layers must be positive. Got %du" % Num_Hidden_Layers;
        assert(Neurons_Per_Layer   > 0), "Neurons_Per_Layer must be positive. Got %u" % Neurons_Per_Layer;
        assert(Input_Dim           > 0), "Input_Dim must be positive. Got %u"  % Input_Dim;
        assert(Output_Dim          > 0), "Output_Dim must be positive. Got %u" % Output_Dim;

        super(NN, self).__init__()

        # Define object attributes.
        self.Input_Dim = Input_Dim
        self.Output_Dim = Output_Dim
        self.Num_Hidden_Layers = Num_Hidden_Layers
        self.Batch_Norm = Batch_Norm

        # Initialize the Layers. We hold all layers in a ModuleList.
        self.Layers = torch.nn.ModuleList()

        # Initialize Batch Normalization, if we're doing that.
        if(Batch_Norm == True):
            self.Norm_Layer = torch.nn.BatchNorm1d(
                                    num_features = Input_Dim,
                                    dtype        = Data_Type,
                                    device       = Device)

        # Append the first hidden layer. The domain of this layer is
        # R^{Input_Dim}. Thus, in_features = Input_Dim. Since this is a hidden
        # layer, its co-domain is R^{Neurons_Per_Layer}. Thus, out_features =
        # Neurons_Per_Layer.
        self.Layers.append(torch.nn.Linear(
                                in_features  = Input_Dim,
                                out_features = Neurons_Per_Layer,
                                bias         = True ).to(dtype = Data_Type, device = Device))

        # Now append the rest of the hidden layers. Each maps from
        # R^{Neurons_Per_Layer} to itself. Thus, in_features = out_features =
        # Neurons_Per_Layer. We start at i = 1 because we already created the
        # 1st hidden layer.
        for i in range(1, Num_Hidden_Layers):
            self.Layers.append(torch.nn.Linear(
                                    in_features  = Neurons_Per_Layer,
                                    out_features = Neurons_Per_Layer,
                                    bias         = True ).to(dtype = Data_Type, device = Device))

        # Now, append the Output Layer, which has Neurons_Per_Layer input
        # features, but only Output_Dim output features.
        self.Layers.append(torch.nn.Linear(
                                in_features  = Neurons_Per_Layer,
                                out_features = Output_Dim,
                                bias         = True ).to(dtype = Data_Type, device = Device))

        # Initialize the weight matrices, bias vectors in the network.
        if(Activation_Function == "Tanh" or Activation_Function == "Rational"):
            Gain = 0
            if  (Activation_Function == "Tanh"):
                Gain = 5./3.
            elif(Activation_Function == "Rational"):
                Gain = 1.41

            for i in range(self.Num_Hidden_Layers + 1):
                torch.nn.init.xavier_normal_(self.Layers[i].weight, gain = Gain)
                torch.nn.init.zeros_(self.Layers[i].bias)

        elif(Activation_Function == "Sin"):
            # The SIREN paper suggests initializing the elements of every weight
            # matrix (except for the first one) by sampling a uniform
            # distribution over [-c/root(n), c/root(n)], where c > root(6),
            # and n is the number of neurons in the layer. I use c = 3 > root(6).
            #
            # Further, for simplicity, I initialize each bias vector to be zero.
            a = 3./math.sqrt(Neurons_Per_Layer)
            for i in range(0, self.Num_Hidden_Layers + 1):
                torch.nn.init.uniform_( self.Layers[i].weight, -a, a)
                torch.nn.init.zeros_(   self.Layers[i].bias)

        # Finally, set the Network's activation functions.
        self.Activation_Functions = torch.nn.ModuleList()
        if  (Activation_Function == "Tanh"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.Tanh())
        elif(Activation_Function == "Sin"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Sin())
        elif(Activation_Function == "Rational"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Rational(Data_Type = Data_Type, Device = Device))
        else:
            print("Unknown Activation Function. Got %s" % Activation_Function)
            print("Thrown by Neural_Network.__init__. Aborting.")
            exit();



    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ Forward method for the NN class. Note that the user should NOT call
        this function directly. Rather, they should call it through the __call__
        method (using the NN object like a function), which is part of the
        module class and calls forward.

        ------------------------------------------------------------------------
        Arguments:

        X: A batch of inputs. This should be a B by Input_Dim tensor, where B
        is the batch size. The ith row of X should hold the ith input.

        ------------------------------------------------------------------------
        Returns:

        If X is a B by Input_Dim tensor, then the output of this function is a
        B by Output_Dim tensor, whose ith row holds the value of the network
        applied to the ith row of X. """

        # If we are using batch normalization, then normalize the inputs.
        if(self.Batch_Norm == True):
            X = self.Norm_Layer(X);

        # Pass X through the hidden layers. Each has an activation function.
        for i in range(0, self.Num_Hidden_Layers):
            X = self.Activation_Functions[i](self.Layers[i](X));

        # Pass through the last layer (with no activation function) and return.
        return self.Layers[self.Num_Hidden_Layers](X);


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_text):
    if device_text:
        device = torch.device(device_text)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested, but torch.cuda.is_available() is False.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_case_dataset(case_path, field_name="C_observed"):
    payload = np.load(case_path)
    if field_name not in payload:
        raise KeyError("{} not found in {}".format(field_name, case_path))

    C = np.asarray(payload[field_name], dtype=np.float32)
    x = np.asarray(payload["x"], dtype=np.float32)
    t = np.asarray(payload["t"], dtype=np.float32)
    nt, nx = C.shape

    tt, xx = np.meshgrid(t, x, indexing="ij")
    inputs = np.column_stack([xx.reshape(-1), tt.reshape(-1)]).astype(np.float32)
    targets = C.reshape(-1, 1).astype(np.float32)

    variables = {}
    for key in ["U", "L", "D", "k", "Pe", "Da"]:
        if key in payload:
            variables[key] = float(np.asarray(payload[key]))

    return inputs, targets, variables, nx, nt


def split_train_val(inputs, targets, val_fraction, seed):
    n_samples = inputs.shape[0]
    indices = np.arange(n_samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = int(round(n_samples * val_fraction))
    n_val = min(max(n_val, 1), n_samples - 1)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return inputs[train_idx], targets[train_idx], inputs[val_idx], targets[val_idx]


def make_loader(x_values, y_values, batch_size, shuffle):
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            batch_size = xb.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


def evaluate_full_batch(model, x_val, y_val, criterion):
    model.eval()
    with torch.no_grad():
        pred = model(x_val)
        loss = criterion(pred, y_val)
    return float(loss.item())


def save_checkpoint(
    path,
    model,
    args,
    case_name,
    variables,
    epoch,
    train_loss,
    val_loss,
):
    torch.save(
        {
            "case_name": case_name,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "class_name": "NN",
                "Num_Hidden_Layers": args.layers,
                "Neurons_Per_Layer": args.hidden,
                "Input_Dim": 2,
                "Output_Dim": 1,
                "Activation_Function": ACTIVATION_FUNCTION,
                "Batch_Norm": False,
            },
            "input_columns": ["x", "t"],
            "output_column": "c",
            "input_normalization": None,
            "variables": variables,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        },
        path,
    )


def train_one_case(case_path, args, device):
    case_name = os.path.splitext(os.path.basename(case_path))[0]
    inputs, targets, variables, nx, nt = load_case_dataset(case_path, field_name=args.field)
    print("{} | nx={} | nt={} | total_points={}".format(case_name, nx, nt, inputs.shape[0]))

    x_train, y_train, x_val, y_val = split_train_val(
        inputs,
        targets,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        "{} | train data num={} | valid data num={}".format(
            case_name,
            x_train.shape[0],
            x_val.shape[0],
        )
    )

    use_fast_path = bool(args.fast_full_batch) and x_train.shape[0] < args.fast_full_batch_threshold
    if use_fast_path:
        print(
            "{} | fast full-batch training enabled because train data num < {}".format(
                case_name,
                args.fast_full_batch_threshold,
            )
        )
        x_train_full = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_full = torch.tensor(y_train, dtype=torch.float32, device=device)
        x_val_full = torch.tensor(x_val, dtype=torch.float32, device=device)
        y_val_full = torch.tensor(y_val, dtype=torch.float32, device=device)
        train_loader = None
        val_loader = None
    else:
        train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
        val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    model = NN(
        Num_Hidden_Layers=args.layers,
        Neurons_Per_Layer=args.hidden,
        Input_Dim=2,
        Output_Dim=1,
        Data_Type=torch.float32,
        Device=device,
        Activation_Function=ACTIVATION_FUNCTION,
        Batch_Norm=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = -1

    model_path = os.path.join(MODEL_DIR, "{}_best.pt".format(case_name))

    for epoch in range(1, args.epochs + 1):
        model.train()
        if use_fast_path:
            optimizer.zero_grad()
            pred = model(x_train_full)
            loss = criterion(pred, y_train_full)
            loss.backward()
            optimizer.step()
            train_loss = float(loss.item())
        else:
            train_loss_sum = 0.0
            train_count = 0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                batch_size = xb.shape[0]
                train_loss_sum += float(loss.item()) * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(train_count, 1)

        should_validate = (epoch % args.valid_every == 0) or (epoch == args.epochs)
        if should_validate:
            if use_fast_path:
                val_loss = evaluate_full_batch(model, x_val_full, y_val_full, criterion)
            else:
                val_loss = evaluate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_epoch = epoch
                save_checkpoint(
                    model_path,
                    model,
                    args,
                    case_name,
                    variables,
                    epoch,
                    train_loss,
                    val_loss,
                )

            print(
                "{} | epoch {:04d} | train_loss={:.6e} | val_loss={:.6e} | best={:.6e} | best_epoch={}".format(
                    case_name,
                    epoch,
                    train_loss,
                    val_loss,
                    best_val_loss,
                    best_epoch,
                )
            )
        elif epoch == 1 or epoch % args.print_every == 0:
            print("{} | epoch {:04d} | train_loss={:.6e}".format(case_name, epoch, train_loss))

    return {
        "case_name": case_name,
        "model_path": model_path,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
    }


def find_case_files(data_dir, tag=""):
    pattern = os.path.join(data_dir, "case_*.npz")
    matcher = case_npz_regex(tag)
    paths = sorted(glob.glob(pattern))
    return [path for path in paths if matcher.match(os.path.basename(path))]


def save_training_summary(records, tag=""):
    summary_path = os.path.join(MODEL_DIR, add_noise_suffix("training_summary.csv", tag))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("case_name,model_path,best_epoch,best_train_loss,best_val_loss,n_train,n_val\n")
        for item in records:
            f.write(
                "{case_name},{model_path},{best_epoch},{best_train_loss:.10e},{best_val_loss:.10e},{n_train},{n_val}\n".format(
                    **item
                )
            )

    json_path = os.path.join(MODEL_DIR, add_noise_suffix("training_summary.json", tag))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return summary_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one ANN surrogate model for every reaction-advection-diffusion case."
    )
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory containing case_*.npz files.")
    parser.add_argument("--field", default="C_observed", choices=["C_observed", "C_clean"])
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--fast_full_batch", action="store_true", default=True)
    parser.add_argument("--no_fast_full_batch", action="store_false", dest="fast_full_batch")
    parser.add_argument("--fast_full_batch_threshold", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--valid_every", type=int, default=500)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=525)
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:0. Default: cuda.")
    add_noise_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.valid_every <= 0:
        raise ValueError("--valid_every must be a positive integer.")
    if args.print_every <= 0:
        raise ValueError("--print_every must be a positive integer.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(device)
    os.makedirs(MODEL_DIR, exist_ok=True)

    tag = noise_tag(parse_noise_level(args.noise_level))
    case_files = find_case_files(args.data_dir, tag=tag)
    if not case_files:
        raise FileNotFoundError(
            "No case_*.npz files found in {} for noise tag '{}'".format(
                args.data_dir,
                tag or "clean",
            )
        )

    print("Use device: {}".format(device))
    print("noise_level={} tag={}".format(parse_noise_level(args.noise_level), tag or "clean"))
    print("Found {} case datasets in {}".format(len(case_files), args.data_dir))
    print("Models will be saved to {}".format(MODEL_DIR))

    records = []
    for case_path in case_files:
        print("\nTraining {}".format(os.path.basename(case_path)))
        records.append(train_one_case(case_path, args, device))

    summary_path, json_path = save_training_summary(records, tag=tag)
    print("\nTraining finished.")
    print("Summary saved:")
    print(summary_path)
    print(json_path)


if __name__ == "__main__":
    main()
