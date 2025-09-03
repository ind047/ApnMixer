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
from datetime import datetime

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
    from datetime import datetime

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

    # Get trial ID from environment or generate one
    # Ray Tune sets this environment variable in worker processes
    trial_id = os.environ.get("TUNE_TRIAL_ID", None)
    if trial_id is None:
        # Fallback: generate a unique trial ID
        trial_id = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    trial_id = f"train_model_{trial_id}"  # Prefix for clarity in logs

    trial_log_path = os.path.join("logs", f"{trial_id}.log")
    best_log_path = os.path.join("logs", "best_so_far.log")

    # Global best tracker (shared across all trials)
    global_best_path = os.path.join("logs", "global_best.json")

    # Initialize global best if it doesn't exist
    if not os.path.exists(global_best_path):
        with open(global_best_path, "w") as f:
            json.dump({"best_mse": float("inf"), "best_config": None}, f)

    # --- Setup ---
    args = config["args"]
    args.lr = config["lr"]
    args.w_decay = config["w_decay"]
    args.nlayer = config["nlayer"]
    args.hid_dim = config["hid_dim"]
    args.expansion_factor = config["expansion_factor"]
    args.batch_size = config["batch_size"]
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

    # Write trial start info
    with open(trial_log_path, "w") as f:
        f.write(f"=== Trial {trial_id} Started ===\n")
        f.write(f"Start time: {datetime.now()}\n")
        f.write(
            f"Config: {json.dumps({k: v for k, v in config.items() if k != 'args'}, indent=2)}\n"
        )
        f.write("=" * 50 + "\n")

    # --- Training Loop ---
    for epoch in range(args.epoch):
        epoch_start = datetime.now()

        model.train()
        train_loss = 0.0

        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch_dict)
            train_res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += train_res["loss"].item()

        avg_train_loss = train_loss / num_batches

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_res = evaluation(
                model, data_obj["val_dataloader"], data_obj["n_val_batches"]
            )
        val_mse = val_res["mse"]

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        # Write to trial log
        with open(trial_log_path, "a") as f:
            f.write(
                f"Epoch {epoch:3d} | train_loss: {avg_train_loss:.6f} | val_mse: {val_mse:.6f} | time: {epoch_time:.2f}s | {datetime.now()}\n"
            )

        # Check and update global best (thread-safe)
        try:
            # Read current global best
            with open(global_best_path, "r") as f:
                global_best = json.load(f)

            current_global_best_mse = global_best.get("best_mse", float("inf"))

            # If this is a new global best
            if val_mse < current_global_best_mse:
                new_best = {
                    "best_mse": val_mse,
                    "best_config": {
                        "trial_id": trial_id,
                        "epoch": epoch,
                        "timestamp": datetime.now().isoformat(),
                        "config": {
                            "lr": args.lr,
                            "w_decay": args.w_decay,
                            "batch_size": args.batch_size,
                            "nlayer": args.nlayer,
                            "hid_dim": args.hid_dim,
                            "expansion_factor": args.expansion_factor,
                            "use_attention": args.use_attention,
                        },
                    },
                }

                # Write new global best
                with open(global_best_path, "w") as f:
                    json.dump(new_best, f, indent=2)

                # Also append to best log for history
                with open(best_log_path, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} | NEW BEST: {val_mse:.6f} | Trial: {trial_id} | Epoch: {epoch} | Config: {json.dumps(new_best['best_config']['config'])}\n"
                    )

                # Log in trial file too
                with open(trial_log_path, "a") as f:
                    f.write(f"*** NEW GLOBAL BEST! MSE: {val_mse:.6f} ***\n")

        except Exception as e:
            # If there's an issue with file operations, continue training
            with open(trial_log_path, "a") as f:
                f.write(f"Error updating global best: {e}\n")

        # Report metrics to Ray Tune
        tune.report({"val_mse": val_mse, "epoch": epoch, "train_loss": avg_train_loss})

    # Write trial completion info
    with open(trial_log_path, "a") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Trial completed at: {datetime.now()}\n")
        f.write(f"Final validation MSE: {val_mse:.6f}\n")


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

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Initialize main log
    main_log_path = os.path.join("logs", "tune_main.log")
    with open(main_log_path, "w") as f:
        f.write(f"=== Ray Tune Hyperparameter Search Started ===\n")
        f.write(f"Start time: {datetime.now()}\n")
        f.write(f"Total trials: 50\n")
        f.write(f"Max epochs per trial: {args.epoch}\n")
        f.write("=" * 60 + "\n")

    # --- Ray Tune Search Space ---
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "w_decay": tune.loguniform(1e-5, 1e-3),
        "nlayer": tune.choice([1, 2, 4]),
        "hid_dim": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([64, 128, 256]),
        "expansion_factor": tune.choice([1, 2, 4]),
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

    # Final results
    print("Best hyperparameters found were: ", analysis.best_config)
    print(f"Best validation MSE: {analysis.best_result['val_mse']:.6f}")

    # Write final summary
    with open(main_log_path, "a") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Tuning completed at: {datetime.now()}\n")
        f.write(f"Best validation MSE: {analysis.best_result['val_mse']:.6f}\n")
        f.write(f"Best config: {json.dumps(analysis.best_config, indent=2)}\n")

    # Read and display final global best
    try:
        global_best_path = os.path.join("logs", "global_best.json")
        if os.path.exists(global_best_path):
            with open(global_best_path, "r") as f:
                final_best = json.load(f)
            print(f"\nFinal Global Best MSE: {final_best['best_mse']:.6f}")
            print(f"From Trial: {final_best['best_config']['trial_id']}")
    except Exception as e:
        print(f"Could not read global best: {e}")
