"""Configuration file for the handover environment."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Configuration class for the handover environment.

    This class contains the configuration parameters for the handover environment,
    3GPP handover protocol, and PPO training.
    """

    # Data parameters
    delta_t_ms: int = 10  # Time step in ms (delta t)

    clip_sinr: bool = True  # Clip SINR values
    sinr_upper_clip: float = 10.0  # Upper clipping value for SINR in dB
    sinr_lower_clip: float = -10.0  # Lower clipping value for SINR in dB

    l3_k: int = 16  # L3 filter coefficient
    l3_filter_w: float = 1 / (2 ** (l3_k / 4))  # L3 filter weight

    # Network
    fc: float = 2.1e9  # Carrier frequency in Hz
    bw: float = 10e6  # bandwidth in Hz

    # Handover environment parameters
    q_in_db: float = -6.0  # SINR-QoS thresholds ind dB
    q_out_db: float = -8.0
    t_ho_prep: int = 50 // delta_t_ms  # HO preparation time
    t_ho_exec: int = 40 // delta_t_ms  # HO execution time
    t_mts: int = 1_000 // delta_t_ms  # Minimum-time-of-stay (MTS)
    t_rlfr: int = 200 // delta_t_ms  # Avg time for RLF recovery
    t_t310: int = 1_000 // delta_t_ms  # T310 timer
    n310 = 10  # Counter for starting T310 timer
    n311 = 3  # Counter for stopping T310 timer

    # 3GPP parameters
    a3_hys: float = 1.0  # Hysteresis for A3
    a3_ttt_ms: int = 40 // delta_t_ms  # TTT for A3
    a3_off: float = 1.0

    # PPO parameters
    rew_const: float = 0.95  # reward constant
    net_arch: list[int] = field(default_factory=lambda: [64, 128, 64])

    # Training parameters
    use_wandb: bool = False  # Use Weights & Biases for logging
    train_or_test = "test"
    n_steps_total: int = int(5e6)
    n_steps_per_update: int = 2000
    batch_size: int = 200
    n_epochs: int = 10
    lr: float = 5e-5
    ent_coef: float = 0.1

    # Environment parameters
    terminate_on_pp: bool = True  # Terminate episode on ping-pong
    terminate_on_rlf: bool = True  # Terminate episode on RLF
    test_deterministic_actions: bool = True  # Test with deterministic actions
    permit_ho_prep_abort: bool = True  # Permit HO preparation abort

    def update(self, config_dict: dict):
        """
        Update the configuration with values from a dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration parameters.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config has no attribute '{key}'")
