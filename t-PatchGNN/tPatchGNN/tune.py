import os
import sys
import argparse
import torch
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import threading
import json

# Fix the path setup for Ray workers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import after path is set
import lib.utils as utils
from lib.parse_datasets import parse_datasets
from lib.evaluation import compute_all_losses, evaluation
from model.APNTSMixer import APNTSMixer


def train_model(config):
    """Trainable function for Ray Tune"""
    # Re-setup path in worker (critical for Ray)
    import os
    import sys
    import torch
    import torch.optim as optim
    import json

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    import lib.utils as utils
    from lib.parse_datasets import parse_datasets
    from lib.evaluation import compute_all_losses, evaluation
    from model.APNTSMixer import APNTSMixer

    # --- Logging setup ---
    os.makedirs("logs", exist_ok=True)
    trial_id = tune.get_trial_id()
    trial_log_path = os.path.join("logs", f"trial_{trial_id}.log")
    best_log_path = os.path.join("logs", "best_so_far.log")
    best_lock = threading.Lock()  # For thread-safe writing

    # --- Setup ---
    args = config["args"]
    args.lr = config["lr"]
    args.w_decay = config["w_decay"]
    args.nlayer = config["nlayer"]
    args.hid_dim = config["hid_dim"]
    args.expansion_factor = config["expansion_factor"]
    args.batch_size = config["batch_size"]
    # args.dropout = config["dropout"]
    args.use_attention = config["use_attention"]

    # Ensure model is set correctly
    args.model = "APNTSMixer"

    utils.setup_seed(args.seed)

    # --- Data ---
    data_obj = parse_datasets(args, patch_ts=False)
    input_dim = data_obj["input_dim"]
    args.ndim = input_dim

    # --- Model ---
    model = APNTSMixer(args).to(args.device)

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

    num_batches = data_obj["n_train_batches"]

    best_mse = float("inf")
    best_config = None

    # --- Training Loop ---
    for epoch in range(args.epoch):
        model.train()
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch_dict)
            train_res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_res = evaluation(
                model, data_obj["val_dataloader"], data_obj["n_val_batches"]
            )
        val_mse = val_res["mse"]

        # Write to trial log
        with open(trial_log_path, "a") as f:
            f.write(f"Epoch {epoch} - val_mse: {val_mse}\n")

        # Check and update best-so-far
        if val_mse < best_mse:
            best_mse = val_mse
            best_config = {
                "trial_id": trial_id,
                "epoch": epoch,
                "val_mse": val_mse,
                "config": {
                    "lr": args.lr,
                    "w_decay": args.w_decay,
                    "batch_size": args.batch_size,
                    "nlayer": args.nlayer,
                    "hid_dim": args.hid_dim,
                    "expansion_factor": args.expansion_factor,
                    # "dropout": args.dropout,
                    "use_attention": args.use_attention,
                },
            }
            # Write best-so-far to shared log (thread-safe)
            with best_lock:
                with open(best_log_path, "a") as bf:
                    bf.write(json.dumps(best_config) + "\n")

        # Report metrics to Ray Tune - FIX: Pass as dictionary
        tune.report({"val_mse": val_mse, "epoch": epoch})


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ray Tune for APNTSMixer")

    # Core arguments
    parser.add_argument("--state", type=str, default="def")
    parser.add_argument("-n", type=int, default=int(1e8))
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--patch_size", type=float, default=24)
    parser.add_argument("--stride", type=float, default=24)
    # parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save", type=str, default="experiments/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="physionet")
    parser.add_argument("--quantization", type=float, default=0.0)
    parser.add_argument("--te_dim", type=int, default=10)
    parser.add_argument("--node_dim", type=int, default=10)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--t_obs", type=int, default=None)
    parser.add_argument("--npatch", type=int, default=20)

    # Missing arguments that APNTSMixer might need
    parser.add_argument("--model", type=str, default="APNTSMixer")
    parser.add_argument("--outlayer", type=str, default="Linear")
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--tf_layer", type=int, default=1)
    parser.add_argument("--logmode", type=str, default="a")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--use_end_attention", type=bool, default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.t_obs is None:
        args.t_obs = args.history

    # --- Ray Tune Search Space ---
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "w_decay": tune.loguniform(1e-5, 1e-3),
        "nlayer": tune.choice([1, 2, 4]),
        "hid_dim": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([64, 128, 256]),
        "expansion_factor": tune.choice([1, 2, 4]),
        # "dropout": tune.uniform(0.1, 0.5),
        "use_attention": tune.choice([True, False]),
        "args": args,
    }

    # --- Run Tuning ---
    scheduler = ASHAScheduler(
        metric="val_mse",
        mode="min",
        max_t=args.epoch,
        grace_period=10,
        reduction_factor=2,
    )

    analysis = tune.run(
        train_model,
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
        config=search_space,
        num_samples=50,
        scheduler=scheduler,
        name="apntsmixer_tuning",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
