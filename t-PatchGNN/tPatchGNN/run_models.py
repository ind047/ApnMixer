import os
import sys

sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import lib.utils as utils
from lib.parse_datasets import parse_datasets

# Ray Tune imports
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray Tune not available. Install with: pip install ray[tune]")

from lib.evaluation import compute_all_losses, evaluation
from model.tPatchGNN import tPatchGNN
from model.APN import tAPN
from model.APNTSMixer import APNTSMixer

parser = argparse.ArgumentParser("IMTS Forecasting")

parser.add_argument("--state", type=str, default="def")
parser.add_argument("-n", type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument("--hop", type=int, default=1, help="hops in GNN")
parser.add_argument("--nhead", type=int, default=1, help="heads in Transformer")
parser.add_argument("--tf_layer", type=int, default=1, help="# of layer in Transformer")
parser.add_argument("--nlayer", type=int, default=1, help="# of layer in TSmodel")
parser.add_argument(
    "--epoch", type=int, default=200, help="training epochs (updated default)"
)
parser.add_argument(
    "--patience", type=int, default=10, help="early stopping patience (updated default)"
)
parser.add_argument(
    "--history",
    type=int,
    default=24,
    help="number of hours (months for ushcn and ms for activity) as historical window (used by tPatchGNN only)",
)
parser.add_argument(
    "-ps", "--patch_size", type=float, default=24, help="window size for a patch"
)
parser.add_argument(
    "--stride", type=float, default=24, help="period stride for patch sliding"
)
parser.add_argument("--logmode", type=str, default="a", help="File mode of logging.")

parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
    help="Starting learning rate (updated default 1e-2).",
)
parser.add_argument("--w_decay", type=float, default=0.0, help="weight decay.")
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=256,
    help="Training batch size (updated default 256).",
)

parser.add_argument(
    "--save", type=str, default="experiments/", help="Path for save checkpoints"
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="ID of the experiment to load for evaluation. If None, run a new experiment.",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument(
    "--dataset",
    type=str,
    default="physionet",
    help="Dataset to load. Available: physionet, mimic, ushcn",
)
parser.add_argument(
    "--use_attention",
    action="store_true",  # Better for boolean flags
    help="Whether to use attention mechanism in the model.",
)
# value 0 means using original time granularity, Value 1 means quantization by 1 hour,
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument(
    "--quantization",
    type=float,
    default=0.0,
    help="Quantization on the physionet dataset.",
)
parser.add_argument(
    "--model",
    type=str,
    default="tPatchGNN",
    help="Model name",
    choices=["tPatchGNN", "tAPN", "APNTSMixer"],
)
parser.add_argument("--outlayer", type=str, default="Linear", help="Model name")
parser.add_argument(
    "-hd", "--hid_dim", type=int, default=64, help="Number of units per hidden layer"
)
parser.add_argument(
    "-td", "--te_dim", type=int, default=10, help="Number of units for time encoding"
)
parser.add_argument(
    "-nd", "--node_dim", type=int, default=10, help="Number of units for node vectors"
)
parser.add_argument("--gpu", type=str, default="0", help="which gpu to use.")
parser.add_argument(
    "--t_obs",
    type=int,
    default=None,
    help="observation window size (auto-calculated for tAPN, manual for debugging)",
)
parser.add_argument(
    "--npatch",
    type=int,
    default=None,
    help="Number of patches (for tAPN) or auto-calculated (for tPatchGNN)",
)

# Ray Tune parameters
parser.add_argument(
    "--use_ray_tune",
    action="store_true",
    help="Enable Ray Tune hyperparameter optimization",
)
parser.add_argument(
    "--tune_samples",
    type=int,
    default=50,
    help="Number of hyperparameter combinations to try",
)
parser.add_argument(
    "--tune_epochs",
    type=int,
    default=30,
    help="Maximum epochs per trial",
)
parser.add_argument(
    "--tune_grace_period",
    type=int,
    default=5,
    help="Minimum epochs before early stopping",
)
parser.add_argument(
    "--tune_reduction_factor",
    type=int,
    default=2,
    help="Reduction factor for halving scheduler",
)

args = parser.parse_args()

# Handle npatch calculation differently for different models
if args.model in ["tAPN", "APNTSMixer"]:
    # For tAPN: use specified npatch or default to 20 adaptive patches (updated clarified default)
    if args.npatch is None:
        args.npatch = 20  # Default adaptive patches
else:
    # For tPatchGNN: calculate npatch from patch_size and stride (original behavior)
    if args.npatch is None:
        args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################


def train_apn_tsmixer_with_tune(config, base_args):
    """
    Trainable function for Ray Tune hyperparameter optimization of APNTSMixer.
    """
    # Merge config with base args
    args = argparse.Namespace(**vars(base_args))

    # Update args with Ray Tune config
    for key, value in config.items():
        setattr(args, key, value)

    # Setup device and seed
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.setup_seed(args.seed)

    # Load dataset
    if args.model in ["tAPN", "APNTSMixer"]:
        if args.t_obs is None:
            args.t_obs = args.history
        data_obj = parse_datasets(args, patch_ts=False)
    else:
        if args.t_obs is None:
            args.t_obs = args.history
        data_obj = parse_datasets(args, patch_ts=True)

    input_dim = data_obj["input_dim"]
    args.ndim = input_dim

    # Initialize model
    if args.model == "APNTSMixer":
        model = APNTSMixer(args).to(args.device)
    elif args.model == "tAPN":
        model = tAPN(args).to(args.device)
    elif args.model == "tPatchGNN":
        model = tPatchGNN(args).to(args.device)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )

    num_batches = data_obj["n_train_batches"]
    best_val_mse = np.inf
    best_epoch = 0

    # Training loop
    for epoch in range(args.tune_epochs):
        # Training
        model.train()
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch_dict)
            train_res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_res = evaluation(
                model, data_obj["val_dataloader"], data_obj["n_val_batches"]
            )

        scheduler.step(val_res["mse"])

        # Track best validation performance
        if val_res["mse"] < best_val_mse:
            best_val_mse = val_res["mse"]
            best_epoch = epoch

        # Report to Ray Tune
        tune.report(
            mse=val_res["mse"],
            mae=val_res["mae"],
            rmse=val_res["rmse"],
            mape=val_res["mape"],
            loss=val_res["loss"],
            epoch=epoch,
            best_mse=best_val_mse,
            best_epoch=best_epoch,
        )

        # Early stopping for this trial
        if epoch - best_epoch >= args.patience:
            break


def get_apn_tsmixer_search_space():
    """
    Define the hyperparameter search space for APNTSMixer optimization.
    """
    search_space = {
        # Learning rate
        "lr": tune.loguniform(1e-4, 1e-1),
        # Weight decay
        "w_decay": tune.uniform(0.0, 0.1),
        # Hidden dimension
        "hid_dim": tune.choice([32, 64, 96, 128]),
        # Number of layers
        "nlayer": tune.choice([1, 2, 3, 4]),
        # Number of patches for adaptive patching
        "npatch": tune.choice([16, 20, 24, 40]),
        # Batch size
        "batch_size": tune.choice([64, 128, 256]),
        # Attention mechanism flag
        "use_attention": tune.choice([True, False]),
        # Patience for early stopping
        "patience": tune.choice([5, 8, 10, 12]),
    }

    return search_space


def run_ray_tune_optimization(args):
    """
    Run Ray Tune hyperparameter optimization using the updated API.
    """
    if not RAY_AVAILABLE:
        print("Ray Tune is not available. Please install with: pip install ray[tune]")
        return

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Get search space
    search_space = get_apn_tsmixer_search_space()

    # Configure scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="mse",
        mode="min",
        max_t=args.tune_epochs,
        grace_period=args.tune_grace_period,
        reduction_factor=args.tune_reduction_factor,
    )

    # Configure search algorithm
    try:
        search_alg = OptunaSearch(metric="mse", mode="min")
        print("✅ Using Optuna search algorithm")
    except Exception as e:
        print(f"⚠️ Optuna not available, using default search: {e}")
        search_alg = None

    # Use the newer Tuner API
    print(f"Starting Ray Tune optimization with {args.tune_samples} trials...")
    print(f"Search space: {search_space}")

    # Create the tuner with updated API
    tuner = tune.Tuner(
        tune.with_parameters(train_apn_tsmixer_with_tune, base_args=args),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.tune_samples,
        ),
        param_space=search_space,
        run_config=ray.air.RunConfig(
            name=f"apn_tsmixer_tune_{args.dataset}",
            storage_path=os.path.abspath("ray_results"),  # Use absolute path
            stop={"training_iteration": args.tune_epochs},
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=0,  # Disable checkpointing to save space
                checkpoint_at_end=False,
            ),
        ),
    )

    # Run the optimization
    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result("mse", "min")

    print("\n" + "=" * 60)
    print("RAY TUNE OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation MSE: {best_result.metrics['mse']:.6f}")
    print(f"Best trial final validation MAE: {best_result.metrics['mae']:.6f}")
    print(f"Best trial final validation RMSE: {best_result.metrics['rmse']:.6f}")
    print(f"Best trial final validation MAPE: {best_result.metrics['mape'] * 100:.2f}%")
    print(f"Best trial reached epoch: {best_result.metrics['epoch']}")
    print("=" * 60)

    # Generate command line for best config
    best_config = best_result.config
    cmd_parts = [
        "python run_models.py",
        f"--dataset {args.dataset}",
        f"--model APNTSMixer",
        f"--epoch 200",  # Full training
    ]

    for key, value in best_config.items():
        if key == "use_attention":
            if value:
                cmd_parts.append(f"--{key}")  # For boolean flags, just add the flag
        else:
            cmd_parts.append(f"--{key} {value}")

    best_command = " ".join(cmd_parts)
    print(f"\n🚀 Command to run best configuration:")
    print(best_command)

    # Save best config to file
    best_config_path = f"best_config_{args.dataset}_{args.model}.txt"
    with open(best_config_path, "w") as f:
        f.write("Best hyperparameter configuration:\n")
        f.write("=" * 40 + "\n")
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nFinal validation MSE: {best_result.metrics['mse']:.6f}\n")
        f.write(f"Final validation MAE: {best_result.metrics['mae']:.6f}\n")
        f.write(f"Final validation RMSE: {best_result.metrics['rmse']:.6f}\n")
        f.write(f"Final validation MAPE: {best_result.metrics['mape'] * 100:.2f}%\n")
        f.write(f"\nBest command to run:\n")
        f.write(f"{best_command}\n")

    print(f"Best configuration saved to: {best_config_path}")

    # Shutdown Ray
    ray.shutdown()

    return best_result


#####################################################################################################

if __name__ == "__main__":
    utils.setup_seed(args.seed)

    # Check if Ray Tune optimization is requested
    if args.use_ray_tune:
        if args.model != "APNTSMixer":
            print(
                "Ray Tune optimization is currently only supported for APNTSMixer model."
            )
            print("Please set --model APNTSMixer to use Ray Tune.")
            sys.exit(1)

        print("Starting Ray Tune hyperparameter optimization for APNTSMixer...")

        # Use the updated function (choose one):
        best_result = run_ray_tune_optimization(args)  # Use the fixed version
        # OR
        # best_result = run_ray_tune_optimization_simple(args)  # Use the simpler version

        print("\nRay Tune optimization completed!")
        print("Use the best configuration found above to train your final model.")
        sys.exit(0)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + ".ckpt")

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2) :]
    input_command = " ".join(input_command)

    # utils.makedirs("results/")

    ##################################################################
    # For tAPN: Allow full temporal extent, don't constrain by history
    if args.model in ["tAPN", "APNTSMixer"]:
        # For tAPN, we want to use more of the available temporal data
        # Set history to a larger value to capture more temporal context
        if args.t_obs is None:
            # Set t_obs to allow adaptive patching over a larger window
            # args.t_obs = args.history * 2  # Use 2x the history for adaptive patching
            args.t_obs = args.history
        # # Use a larger history window for tAPN to get more temporal data
        # original_history = args.history
        # args.history = min(args.history * 3, 72)  # Use up to 3x history (max 72 hours)
        data_obj = parse_datasets(args, patch_ts=False)
        # args.history = original_history  # Restore for logging

        # print(
        #     f"tAPN: Using extended temporal window (history={min(original_history * 3, 72)}) with t_obs={args.t_obs} for adaptive patching"
        # )
    else:
        # For tPatchGNN: use standard history-based windowing
        if args.t_obs is None:
            args.t_obs = args.history  # For tPatchGNN, t_obs equals history
        data_obj = parse_datasets(args, patch_ts=True)
    input_dim = data_obj["input_dim"]

    ### Model setting ###
    args.ndim = input_dim
    if args.model == "tPatchGNN":
        model = tPatchGNN(args).to(args.device)
    elif args.model == "tAPN":
        model = tAPN(args).to(args.device)
    elif args.model == "APNTSMixer":
        model = APNTSMixer(args).to(args.device)

    ##################################################################

    # # Load checkpoint and evaluate the model
    # if args.load is not None:
    # 	utils.get_ckpt_model(ckpt_path, model, args.device)
    # 	exit()

    ##################################################################

    if args.n < 12000:
        args.state = "debug"
        log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
    else:
        if args.model == "tAPN":
            log_path = "logs/{}_{}_{}_{}patch_{}layer_{}lr.log".format(
                args.dataset,
                args.model,
                args.state,
                args.npatch,
                args.nlayer,
                args.lr,
            )
        elif args.model == "APNTSMixer":
            log_path = "logs/{}_{}_{}_{}patch_{}layer_{}lr.log".format(
                args.dataset,
                args.model,
                args.state,
                args.npatch,
                args.nlayer,
                args.lr,
            )
        else:
            log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log".format(
                args.dataset,
                args.model,
                args.state,
                args.patch_size,
                args.stride,
                args.nlayer,
                args.lr,
            )

    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(
        logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode
    )
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )

    num_batches = data_obj["n_train_batches"]  # n_sample / batch_size
    print("n_train_batches:", num_batches)

    best_val_mse = np.inf
    test_res = None
    for itr in range(args.epoch):
        st = time.time()

        ### Training ###
        model.train()
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch_dict)
            train_res["loss"].backward()

            # Add gradient clipping here (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        ### Validation ###
        model.eval()
        with torch.no_grad():
            val_res = evaluation(
                model, data_obj["val_dataloader"], data_obj["n_val_batches"]
            )

            ### Testing ###
            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_iter = itr
                test_res = evaluation(
                    model, data_obj["test_dataloader"], data_obj["n_test_batches"]
                )

            # ADD THE SCHEDULER STEP HERE - after validation evaluation
            scheduler.step(val_res["mse"])  # Step based on validation MSE

            logger.info("- Epoch {:03d}, ExpID {}".format(itr, experimentID))
            logger.info(
                "Train - Loss (one batch): {:.5f}".format(train_res["loss"].item())
            )
            logger.info(
                "Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%".format(
                    val_res["loss"],
                    val_res["mse"],
                    val_res["rmse"],
                    val_res["mae"],
                    val_res["mape"] * 100,
                )
            )
            if test_res is not None:
                logger.info(
                    "Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%".format(
                        best_iter,
                        test_res["loss"],
                        test_res["mse"],
                        test_res["rmse"],
                        test_res["mae"],
                        test_res["mape"] * 100,
                    )
                )
            logger.info("Time spent: {:.2f}s".format(time.time() - st))

        if itr - best_iter >= args.patience:
            print("Exp has been early stopped!")
            sys.exit(0)
