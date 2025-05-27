"""Train a PPO agent using the provided configuration."""

from datetime import datetime
import importlib.util
import os

from rl_zoo3 import linear_schedule
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import torch
import wandb

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

SIM_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SWEEP_NAME = "ppo_sweep"
SAVE_MODEL = False


def get_sweep_config():
    """Get the sweep configuration for WandB."""
    return {
        "name": SWEEP_NAME,
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "reward_sum_avg"},
        "parameters": {
            "ent_coef": {"values": [0.001, 0.01, 0.1]},
            "rew_const": {"values": [0.8, 0.9, 1.0]},
        },
    }


def main(root_path: str) -> int:
    """Main function to train or sweep PPO on the handover environment."""
    config = Config()
    if config.use_wandb:
        return sweep_ppo(root_path)
    return train_ppo(root_path)


def sweep_ppo(root_path: str) -> int:
    """Run a WandB sweep for hyperparameter optimization."""
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep=sweep_config, project=SWEEP_NAME)
    wandb.agent(sweep_id, lambda: train_ppo(root_path), count=1)

    return 0


def train_ppo(root_path: str):
    """Train a PPO agent on the handover environment."""
    # Load configuration
    config = Config()

    # Load MATLAB files
    data_dir = os.path.join(root_path, "data", "processed")
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")

    # Speed filter
    use_speed_list = [30, 50]
    rsrp_files, sinr_files, _ = ut.filenames_speed_filter(
        rsrp_files, sinr_files, use_speed_list
    )

    # Load all datasets
    rsrp_list = []
    sinr_list = []
    sinr_norm_list = []
    for rsrp_fname_i, sinr_fname_i in zip(rsrp_files, sinr_files):
        # Load dataset
        rsrp_db, sinr_db = dl.load_preprocess_dataset(
            config, data_dir, rsrp_fname_i, sinr_fname_i
        )

        # Clip and normalize SINR
        if config.clip_sinr:
            sinr_norm = ut.clipnorm(
                sinr_db, config.sinr_lower_clip, config.sinr_upper_clip
            )
        else:
            sinr_norm = sinr_db

        sinr_list.append(sinr_db)
        rsrp_list.append(rsrp_db)
        sinr_norm_list.append(sinr_norm)

    # Generate environment
    env = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    check_env(env, warn=True)

    # WandB
    if config.use_wandb:
        config.update(wandb.config.as_dict())

    # Directories
    if config.use_wandb and wandb.run is not None:
        run_name = f"{wandb.run.name}_{SIM_ID}"
    else:
        run_name = SIM_ID

    model_dir = os.path.join(
        root_path,
        "results",
        "models",
        SWEEP_NAME,
        run_name,
    )
    if importlib.util.find_spec("tensorboard") is not None:
        tensorboard_log_dir = os.path.join(
            root_path,
            "results",
            "tensorboard",
            SWEEP_NAME,
            run_name,
        )
    else:
        tensorboard_log_dir = None

    # PPO model
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=config.net_arch, vf=config.net_arch),
    )
    model = PPO(
        "MlpPolicy",
        env,
        ent_coef=config.ent_coef,
        learning_rate=linear_schedule(config.lr),
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=config.n_steps_per_update,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        tensorboard_log=tensorboard_log_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if SAVE_MODEL:
        model.save(model_dir)

    model.learn(total_timesteps=config.n_steps_total, progress_bar=True)

    if config.use_wandb:
        wandb.finish()

    return 0
