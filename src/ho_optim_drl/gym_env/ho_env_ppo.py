"""Handover environment for training and testing the DRL agent."""

import os
from typing import Any, Optional, TYPE_CHECKING

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .ho_protocol_ppo import HOProcedurePPO

if TYPE_CHECKING:
    from ..config import Config
    import stable_baselines3


class HandoverEnvPPO(gym.Env):
    """
    Handover Environment for the PPO agent.

    This environment is used for training and testing the PPO agent.
    """

    path_to_env = os.path.abspath(__file__)
    env_id: int = 0  # Environment ID

    def __init__(
        self,
        config: "Config",
        rsrp_list: list[np.ndarray],
        sinr_list: list[np.ndarray],
        sinr_norm_list: list[np.ndarray],
    ):
        """Initialize the Handover Environment"""
        super().__init__()

        # Environment parameters
        HandoverEnvPPO.env_id += 1
        self.config = config
        self.test_mode_on = False
        self.terminate_on_rlf = config.terminate_on_rlf
        self.terminate_on_pp = config.terminate_on_pp

        self.ho_procedure = HOProcedurePPO(self.config)

        # Data loader
        self.rsrp_list = rsrp_list
        self.sinr_list = sinr_list
        self.sinr_norm_list = sinr_norm_list

        # Environment parameters
        self.n_datasets = len(rsrp_list)
        self.dataset_idx: int = 0
        self.time_steps, self.n_bs = rsrp_list[0].shape
        self.t = 0

        # Observation space
        self.n_observations = 2 * self.n_bs + 1
        self.o_low = np.zeros(self.n_observations, dtype=np.float32)
        self.o_high = np.ones(self.n_observations, dtype=np.float32)
        self.observation_space = spaces.Box(self.o_low, self.o_high, dtype=np.float32)

        # Action space
        self.action_space = spaces.Discrete(self.n_bs)

        # Observations, flags, etc.
        self.s_action = []

        # RSRP and SINR values
        self.s_rsrp = []
        self.s_sinr = []

        # Cell IDs
        self.s_pcell = []  # Serving cell
        self.s_tcell = []  # Target cell

        # Handover flags
        self.s_ho_complete = []
        self.s_ho_prep = []
        self.s_ho_exec = []
        self.s_q_out_db = []
        self.s_rlf = []
        self.s_pp = []

        # Relative value of counters (t/t_max)
        self.s_rel_n310_t310 = []
        self.s_rel_ho_prep_cnt = []
        self.s_rel_ho_exec_cnt = []
        self.s_rel_mtsc_cnt = []

        # General state
        self.t = 0
        self.state: np.ndarray | None = None
        self.terminated = False
        self.truncated = False

        # Reset environment
        self.reset()

    @classmethod
    def reset_cls(cls):
        """Reset the environment ID counter to 0."""
        cls.env_id = 0

    def set_test_mode(self, test_mode_on):
        """
        Set the test mode on or off: if on, the environment will not terminate on RLF or PP.

        Parameters
        ----------
        test_mode_on : bool
            Test mode flag.
        """
        self.test_mode_on = test_mode_on
        if self.test_mode_on:
            self.terminate_on_pp = False
            self.terminate_on_rlf = False
        else:
            self.terminate_on_pp = self.config.terminate_on_pp
            self.terminate_on_rlf = self.config.terminate_on_rlf

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed, by default None (not used).
        options : dict, optional
            Options for the reset, by default None (not used).

        Returns
        -------
        np.ndarray
            Initial observation.
        """
        # Observations, flags, etc.
        self.s_action = []

        super().reset()

        # RSRP and SINR values
        self.s_rsrp = []
        self.s_rsrp_unscaled = []
        self.s_rsrp_rel_to_pcell = []
        self.s_sinr = []
        self.s_sinr_unscaled = []

        # Cell IDs
        self.s_pcell = []
        self.s_tcell = []
        self.s_connected_idxs = []

        # Handover flags
        self.s_ho_complete = []
        self.s_ho_prep = []
        self.s_ho_exec = []
        self.s_q_out_db = []
        self.s_rlf = []
        self.s_pp = []

        # Relative value of counters (t/t_max)
        self.s_rel_n310_t310 = []
        self.s_rel_ho_prep_cnt = []
        self.s_rel_ho_exec_cnt = []
        self.s_rel_mtsc_cnt = []

        # General state
        self.t = 0
        self.terminated = False
        self.truncated = False

        # Reset the handover procedure
        self.ho_procedure = HOProcedurePPO(self.config)

        # Return the initial observation
        return self._get_initial_observation()

    def _get_initial_observation(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Get the initial observation from the environment.

        Returns
        -------
        tuple[np.ndarray, dict]
            Initial observation and empty dictionary (info).
        """
        rsrp = self.rsrp_list[self.dataset_idx][self.t, :]

        # Init cell = cell with highest RSRP
        pcell = np.argmax(rsrp, axis=0)

        # Initialize the environment states
        self.s_action.append(pcell)

        self.s_ho_complete.append(False)
        self.s_ho_prep.append(False)
        self.s_ho_exec.append(False)
        self.s_q_out_db.append(False)
        self.s_rlf.append(False)
        self.s_pp.append(False)

        self.s_rel_n310_t310.append(0.0)
        self.s_rel_ho_prep_cnt.append(0.0)
        self.s_rel_ho_exec_cnt.append(0.0)
        self.s_rel_mtsc_cnt.append(0.0)

        # Initial state
        s_pcell_indicator = np.zeros(self.n_bs)
        s_pcell_indicator[pcell] = 1
        input_rsrp = self.sinr_norm_list[self.dataset_idx][self.t, :]
        s_pp_indicator = np.array([0])
        self.state = np.concatenate((s_pcell_indicator, input_rsrp, s_pp_indicator))

        return np.array(self.state, dtype=np.float32), {}

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment"""
        # Validate the action and initial state
        if isinstance(action, np.ndarray):
            action = action.item()
        self._validate_state_action(action)

        # Take a step in the handover environment/state machine
        rsrp = self.rsrp_list[self.dataset_idx][self.t, :]
        sinr = self.sinr_list[self.dataset_idx][self.t, :]
        raw_obs = self.ho_procedure.step(rsrp, sinr, action)

        # Update the environment state
        self._update_state(action, raw_obs)

        # Get the reward
        reward = self._get_reward()

        # Terminate episode if RLF flag is set
        if self.terminate_on_rlf and raw_obs["rlf"]:
            self.terminated = True

        # Terminate episode if ping-pong flag is set
        if self.terminate_on_pp and raw_obs["pp"]:
            self.terminated = True

        # Truncate episode if max episode length is reached
        if 1 + self.t == self.time_steps - 1:
            self.truncated = True

        self.t += 1

        s_bs = np.zeros(self.n_bs)  # PCell indicator flag
        pcell = self.ho_procedure.rrc.pcell
        if pcell is not None:
            s_bs[pcell] = 1
        s_sinr = self.sinr_norm_list[self.dataset_idx][self.t, :]  # Next RSRP values
        s_pp = np.array([self.ho_procedure.cntr["mtsc"].pending])

        self.state = np.concatenate((s_bs, s_sinr, s_pp))

        return (
            np.array(tuple(self.state), dtype=np.float32),
            reward,
            self.terminated,
            self.truncated,
            {"sinr_lin": 0},
        )

    def _validate_state_action(self, action: int):
        """Validate the action and state before taking a step"""
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

    def _update_state(self, action: int, raw_obs: dict):
        """Update the environment state based on the action and various flags"""
        self.s_action.append(action)

        rsrp = self.rsrp_list[self.dataset_idx][self.t, :]
        sinr = self.sinr_list[self.dataset_idx][self.t, :]

        self.s_rsrp.append(rsrp)
        self.s_sinr.append(sinr)

        pcell = self.ho_procedure.get_pcell()
        tcell = self.ho_procedure.get_tcell()

        if pcell is None:
            self.s_pcell.append(-1)
            if tcell is None:  # Radio link failure
                self.s_tcell.append(-1)
                self.s_connected_idxs.append(np.zeros(self.n_bs))
            else:  # Handover execution
                self.s_tcell.append(int(tcell))
                connected_idxs = np.zeros(self.n_bs)
                connected_idxs[tcell] = 1
                self.s_connected_idxs.append(connected_idxs)
        else:
            self.s_pcell.append(int(pcell))
            connected_idxs = np.zeros(self.n_bs)
            connected_idxs[pcell] = 1
            self.s_connected_idxs.append(connected_idxs)
            if tcell is None:  # Normal operation
                self.s_tcell.append(-1)
            else:  # Handover preparation
                self.s_tcell.append(tcell)

        self.s_ho_complete.append(raw_obs["ho_complete"])
        self.s_ho_prep.append(raw_obs["ho_prep"])
        self.s_ho_exec.append(raw_obs["ho_exec"])
        self.s_q_out_db.append(raw_obs["q_in_db_out"])
        self.s_rlf.append(raw_obs["rlf"])
        self.s_pp.append(raw_obs["pp"])

        self.s_rel_n310_t310.append(raw_obs["n310_t310_rel_cnt"])
        self.s_rel_ho_prep_cnt.append(raw_obs["ho_prep_rel_cnt"])
        self.s_rel_ho_exec_cnt.append(raw_obs["ho_exec_rel_cnt"])
        self.s_rel_mtsc_cnt.append(raw_obs["mtsc_rel_cnt"])

    def _get_state_list(self):
        """Get the state list for the environment"""
        return [
            # Actions
            self.s_action,
            # RSRP and SINR values
            self.s_rsrp,
            self.s_sinr,
            # Cell IDs
            self.s_pcell,
            self.s_tcell,
            self.s_connected_idxs,
            # Handover flags
            self.s_ho_complete,
            self.s_ho_prep,
            self.s_ho_exec,
            self.s_q_out_db,
            self.s_rlf,
            self.s_pp,
            # Relative value of counters (t/t_max)
            self.s_rel_n310_t310,
            self.s_rel_ho_prep_cnt,
            self.s_rel_ho_exec_cnt,
            self.s_rel_mtsc_cnt,
        ]

    def _get_reward(self) -> float:
        """Get the reward for the current state."""
        reward = 0.0

        sinr = self.sinr_list[self.dataset_idx][self.t, :]
        sinr_norm = self.sinr_norm_list[self.dataset_idx][self.t, :]

        best_bs = np.argmax(sinr)

        if self.s_pcell[-1] > 0:  # Connected
            reward += sinr_norm[self.s_pcell[-1]].item()  # SINR-based reward
            if self.s_pcell[-1] == best_bs:  # Bonus for best BS
                reward += self.config.rew_const

        if self.ho_procedure.pp_detected:  # Penalty for ping-pong
            reward -= self.config.rew_const

        if self.ho_procedure.rlf_detected:  # Penalty for RLF
            reward -= 2 * self.config.rew_const
        elif (  # Penalty for out-of-sync (SINR < Qout)
            self.ho_procedure.cntr["n310"].pending
            or self.ho_procedure.cntr["t310"].pending
        ):
            reward -= self.config.rew_const

        return reward

    def set_dataset_idx(self, dataset_idx):
        """Set the dataset index."""
        self.dataset_idx = dataset_idx
        self.time_steps, self.n_bs = self.rsrp_list[dataset_idx].shape
        self.reset()

    def render(self, mode="human"):
        """Render the environment"""

    def get_statistics(self):
        """Get statistics of the environment."""
        return self.ho_procedure.get_statistics(self.sinr_list[self.dataset_idx])

    @property
    def done(self):
        """Check if the episode is done (terminated or truncated)."""
        return self.terminated or self.truncated


def test_ppo_model(
    env: HandoverEnvPPO, model: "stable_baselines3.PPO", dataset_idx: int
) -> int:
    """
    Test the PPO model on the environment.

    Parameters
    ----------
    env : HandoverEnvPPO
        Handover environment.
    model : stable_baselines3.PPO
        PPO model.
    dataset_idx : int
        Dataset index.

    Returns
    -------
    int
        Total episode reward.
    """

    # Set the environment to test mode
    env.set_test_mode(True)
    env.set_dataset_idx(dataset_idx)

    # Reset the environment and get the initial observation
    obs, _ = env.reset()

    truncated = False
    reward_arr = []
    while not truncated:
        # Predict the action
        action, _ = model.predict(
            obs, deterministic=env.config.test_deterministic_actions
        )

        # Take the action in the environment
        obs, reward, _, truncated, _ = env.step(action)

        # Store results
        reward_arr.append(reward)

    return np.sum(reward_arr)
