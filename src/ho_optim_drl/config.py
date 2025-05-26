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
    train_or_test = "test"
    clip_rsrq: bool = True  # Clip RSRQ values
    delta_t_ms: int = 10  # Time step in ms (delta t)
    clip_h: float = 10.0  # Clipping range
    clip_l: float = -10.0
    l3_k: int = 16  # L3 filter coefficient
    l3_filter_w: float = 1 / (2 ** (l3_k / 4))  # L3 filter weight

    # Network
    bw: float = 10e6  # bandwidth in Hz

    # Handover environment parameters
    q_in: float = -6.0  # SINR-QoS thresholds ind dB
    q_out: float = -8.0
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
    rew_const: float = 1.0  # reward constant
    net_arch: list[int] = field(default_factory=lambda: [64, 128, 64])

    # Training parameters
    lr: float = 1e-4

    # Environment parameters
    terminate_on_pp: bool = True  # Terminate episode on ping-pong
    terminate_on_rlf: bool = True  # Terminate episode on RLF
    test_deterministic_actions: bool = True  # Test with deterministic actions
    permit_ho_prep_abort: bool = False  # Permit HO preparation abort
