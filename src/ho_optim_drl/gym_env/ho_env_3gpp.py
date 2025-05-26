"""3GPP Handover Protocol."""

from typing import Any, TYPE_CHECKING

import numpy as np

from .ho_protocol_3gpp import (
    GeneralCounter,
    HOEventHandler,
    RadioResourceControl,
    SyncSignal,
)

if TYPE_CHECKING:
    from ..config import Config


class HandoverEnv3GPP:
    """
    3GPP handover protocol.

    This class simulates the 3GPP handover protocol and includes:
    - Radio link monitoring (RLM)
    - Radio link failure detection and recovery (RLF)
    - A3 Event handler
    - Counter N310
    - Counter N311
    - Timer T310
    - Timer TTT
    """

    def __init__(self, protocol_config: "Config") -> None:
        # Parameters
        self.config = protocol_config
        self.debug = False
        self.verbose = 0

        self.permit_ho_prep_abort = True

        # General tasks
        self.sync_source = SyncSignal(
            self.config.q_in,
            self.config.q_out,
            self.debug,
            self.verbose,
        )
        self.rrc = RadioResourceControl(
            self.config.q_in,
            self.config.q_out,
            self.debug,
            self.verbose,
        )
        self.event_handler = HOEventHandler(
            debug=self.debug,
            kwargs={
                "hys_a3": self.config.a3_hys,
                "off_a3": self.config.a3_off,
                "debug_lvl": self.verbose,
            },
        )
        self.cntr = {
            "n310": GeneralCounter(
                self.config.n310,
                1,
                "N310",
                self.debug,
                self.verbose,
            ),
            "n311": GeneralCounter(
                self.config.n311,
                1,
                "N311",
                self.debug,
                self.verbose,
            ),
            "t310": GeneralCounter(
                self.config.t_t310,
                1,
                "T310",
                self.debug,
                self.verbose,
            ),
            "rlfr": GeneralCounter(
                self.config.t_rlfr,
                1,
                "RLFRecovery",
                self.debug,
                self.verbose,
            ),
            "tttc": GeneralCounter(
                self.config.a3_ttt_ms,
                1,
                "TTT",
                self.debug,
                self.verbose,
            ),
            "mtsc": GeneralCounter(
                self.config.t_mts,
                1,
                "MTS",
                self.debug,
                self.verbose,
            ),
            "ho_prep": GeneralCounter(
                self.config.t_ho_prep,
                1,
                "HOPrep",
                self.debug,
                self.verbose,
            ),
            "ho_exec": GeneralCounter(
                self.config.t_ho_exec,
                1,
                "HOExec",
                self.debug,
                self.verbose,
            ),
        }

        # Timeline
        self.bs_idxs = []
        self.rsrp_timeline = []
        self.sinr_timeline = []
        self.sinr_at_ho_exe_pcell = []
        self.sinr_after_ho_exe_tcell = []

        # Benchmark
        self.max_mean_c = 0.0

    def reset(self, tic: int) -> None:
        """Reset task."""
        self.sync_source.reset(tic)
        self.event_handler.reset(tic)
        self.rrc.reset(tic)
        for task in self.cntr.values():
            task.reset(tic)
        self.bs_idxs = []
        self.rsrp_timeline = []
        self.sinr_timeline = []
        self.sinr_at_ho_exe_pcell = []
        self.sinr_after_ho_exe_tcell = []

    def _step_counter_tasks(self, tic: int) -> None:
        """Step counter tasks."""
        for task in self.cntr.values():
            task.step(tic)

    def _radio_link_monitoring(self, sinr_db: np.ndarray, tic: int) -> None:
        """Radio link monitoring (RLM)."""
        if self.rrc.is_connected:
            # Connected to cell
            if self.sync_source.check_out_of_sync(sinr_db, self.rrc.pcell, tic):
                # Out-of-sync signal, reset N311 counter and start N310 counter
                self.cntr["n311"].reset(tic)
                if not self.cntr["t310"].pending:
                    # T310 not running, start N310 counter
                    self.cntr["n310"].start(tic)
            else:
                # In-sync signal, Reset N310 counter
                self.cntr["n310"].reset(tic)
                if self.cntr["t310"].pending:
                    # T310 running and in-sync signal, start N311 counter
                    self.cntr["n311"].start(tic)
        else:  # Disconnected from cell, reset N310, N311, T310
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)

    def _msr_event_handler(self, rsrp_db: np.ndarray, tic: int) -> None:
        """Event handler."""
        if self.rrc.is_connected:  # Connected to cell
            pcell = self.rrc.pcell
            ncell = self.rrc.ncell
            # Event A3: neighbor cell becomes offset better than SpCell
            if self.event_handler.event_a3(rsrp_db, pcell, ncell, tic):
                # Start TTT if not in progress or in HO preparation
                if not self.cntr["ho_prep"].pending and not self.cntr["tttc"].pending:
                    self.event_handler.target_cell = ncell
                    self.cntr["tttc"].start(tic)
                if ncell != self.event_handler.target_cell:  # Target cell changed
                    if self.cntr["tttc"].pending:  # Reset TTT
                        self.cntr["tttc"].reset(tic)
                    if self.cntr["ho_prep"].pending:  # Cancel HO preparation
                        self.cntr["ho_prep"].reset(tic)
            else:  # A3 leaving condition
                if self.cntr["tttc"].pending:  # Reset TTT
                    self.cntr["tttc"].reset(tic)
                if self.cntr["ho_prep"].pending:  # Cancel HO preparation
                    self.cntr["ho_prep"].reset(tic)
        else:  # Disconnected from cell, reset TTT and HO preparation
            self.cntr["tttc"].reset(tic)
            self.cntr["ho_prep"].reset(tic)

    def _t310_state_machine(self, tic: int) -> None:
        """Timer T310 state machine."""
        if self.cntr["n310"].reached_max:  # N310 max reached, start timer T310
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].start(tic)
        elif self.cntr["n311"].reached_max:  # N311 max reached, reset timer T310
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)
        elif self.cntr["t310"].reached_max:  # T310 expired, declare RLF
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)
            self._rlf_recovery(tic, start=True)  # Start RLF recovery

    def _rlf_recovery(self, tic: int, start: bool = False) -> None:
        """RLF recovery."""
        if start:  # Start RLF recovery
            self.cntr["rlfr"].start(tic)
            self.rrc.disconnect(tic)
        else:
            if self.cntr["rlfr"].reached_max:
                if self.rrc.cell_detected:  # Suitable cell detected, reconnect
                    self.rrc.reconnection(tic)
                    self.cntr["rlfr"].reset(tic)

    def _ho_state_machine(self, sinr_db: np.ndarray, tic: int) -> None:
        """Handover state machine."""
        if self.cntr["tttc"].reached_max:
            # TTT expired, start HO preparation and reset TTT
            self.cntr["tttc"].reset(tic)
            self.cntr["ho_prep"].start(tic)
        elif self.cntr["ho_prep"].reached_max:
            # HO preparation done, reset HO preparation and start HO execution
            self.cntr["ho_prep"].reset(tic)
            self.cntr["ho_exec"].start(tic)
            # Save SINR at HO execution
            self.sinr_at_ho_exe_pcell.append(sinr_db[self.rrc.pcell])
            if self.cntr["t310"].pending:
                # T310 running, declare RLF and start RLF recovery
                self.cntr["ho_exec"].reset(tic)
                self._rlf_recovery(tic, start=True)
            else:
                # T310 not running, start HO execution
                self.cntr["ho_prep"].set_done(tic)
                self.rrc.disconnect(tic)
                self._pp_monitoring(tic, start=True)
        elif self.cntr["ho_exec"].reached_max:
            # HO execution done, reconnect to suitable cell
            target_cell = self.event_handler.target_cell
            self.cntr["ho_exec"].reset(tic)
            # Save SINR at HO execution
            self.sinr_after_ho_exe_tcell.append(sinr_db[target_cell])
            if self.sync_source.check_out_of_sync(sinr_db, target_cell, tic):
                # Target BS not reachable, declare RLF and start RLF recovery
                self._rlf_recovery(tic, start=True)
            else:
                # Connect to target cell
                self.cntr["ho_exec"].set_done(tic)
                self.rrc.reconnection(tic)

    def _pp_monitoring(self, tic: int, start: bool = False) -> None:
        """
        Ping-pong monitoring.

        Number of PPs observed:
            self.mtsc.aborted_idxs
        """
        if start:
            # Start ping-pong monitoring
            self.cntr["mtsc"].reset(tic)
            self.cntr["mtsc"].start(tic)
        else:
            # PP monitoring ongoing
            if self.cntr["mtsc"].reached_max:
                # MTS expired, reset PP monitoring
                self.cntr["mtsc"].reset(tic)

    def predict(
        self,
        rsrp_db_mat: np.ndarray,
        sinr_db_mat: np.ndarray,
        **config: dict[str, Any],
    ) -> float:
        """Simulate handover."""
        if len(config):
            self._update_config(**config)

        # Max capacity (upper bound)
        max_sinr_lin = 10 ** (np.max(sinr_db_mat, axis=0) / 10)
        self.max_mean_c = np.mean(np.log2(1 + max_sinr_lin))

        # Initial state
        self.rrc.cell_search(sinr_db_mat[:, 0], 0)  # Init: cell search
        self.rrc.reconnection(0)  # Init: reconnection to the best cell

        for tic, (rsrp_i_db, sinr_i_db) in enumerate(zip(rsrp_db_mat.T, sinr_db_mat.T)):
            # RRC measurement
            self.rrc.rrc_measurement(rsrp_i_db, tic)  # RRC measurement

            # Timer and counter tasks
            self._step_counter_tasks(tic)

            # Monitoring
            self._radio_link_monitoring(sinr_i_db, tic)  # Radio link monitoring

            # State machines
            self._pp_monitoring(tic)  # Ping-pong monitoring
            self._t310_state_machine(tic)

            # Measurement report event handler
            self._msr_event_handler(rsrp_i_db, tic)

            self._ho_state_machine(sinr_i_db, tic)

            # RLF recovery
            self._rlf_recovery(tic)

            # Results
            self.bs_idxs.append(self.rrc.pcell)
            if self.rrc.is_connected:
                self.rsrp_timeline.append(rsrp_i_db[self.rrc.pcell])
                self.sinr_timeline.append(sinr_i_db[self.rrc.pcell])
            else:
                self.rsrp_timeline.append(np.nan)
                self.sinr_timeline.append(np.nan)

        return self._mean_capacity()

    def _update_config(self, **config: dict[str, Any]) -> None:
        """Update configuration."""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} not found.")
        self.reset(0)

    def _mean_capacity(self) -> float:
        """Mean capacity."""
        sinr_t_lin = 10 ** (np.array(self.sinr_timeline) / 10)
        sinr_t_lin[np.isnan(sinr_t_lin)] = 0
        return np.mean(np.log2(1 + sinr_t_lin)).item() if len(sinr_t_lin) > 0 else 0.0

    def get_log(self) -> dict[str, Any]:
        """Get log."""
        mean_c = self._mean_capacity()
        mean_c_rel = 100 * mean_c / self.max_mean_c if self.max_mean_c > 0 else 0
        rlf_rate = (
            100
            * len(self.cntr["rlfr"].start_idxs)
            / len(self.cntr["ho_exec"].start_idxs)
            if len(self.cntr["ho_exec"].start_idxs) > 0
            else 0
        )
        pp_rate = (
            100
            * len(self.cntr["mtsc"].aborted_idxs)
            / len(self.cntr["ho_exec"].start_idxs)
            if len(self.cntr["ho_exec"].start_idxs) > 0
            else 0
        )
        return {
            "max_capacity": self.max_mean_c,
            "achieved_capacity": mean_c,
            "achieved_capacity_rel": mean_c_rel,
            "num_ttt_started": len(self.cntr["tttc"].start_idxs),
            "num_ttt_finished": len(self.cntr["tttc"].terminated_idxs),
            "num_ttt_abort": len(self.cntr["tttc"].aborted_idxs),
            "num_ho_prep_started": len(self.cntr["ho_prep"].start_idxs),
            "num_ho_prep_terminated": len(self.cntr["ho_prep"].terminated_idxs),
            "num_ho_prep_completed": len(self.cntr["ho_prep"].done_idxs),
            "num_ho_prep_aborted": len(self.cntr["ho_prep"].aborted_idxs),
            "num_ho_prep_failed": len(self.cntr["ho_prep"].failed_idxs),
            "num_ho_prep_error": len(self.cntr["ho_prep"].error_idxs),
            "num_ho_exe_started": len(self.cntr["ho_exec"].start_idxs),
            "num_ho_exe_terminated": len(self.cntr["ho_exec"].terminated_idxs),
            "num_ho_exe_completed": len(self.cntr["ho_exec"].done_idxs),
            "num_ho_exe_aborted": len(self.cntr["ho_exec"].aborted_idxs),
            "num_ho_exe_failed": len(self.cntr["ho_exec"].failed_idxs),
            "num_ho_exe_error": len(self.cntr["ho_exec"].error_idxs),
            "num_rlf": len(self.cntr["rlfr"].start_idxs),
            "rlf_rate": rlf_rate,
            "num_pp": len(self.cntr["mtsc"].aborted_idxs),
            "pp_rate": pp_rate,
        }

    def get_statistics(self, sinr_db_mat: np.ndarray) -> dict[str, Any]:
        """Get statistics."""
        n_base_stations, n_samples = sinr_db_mat.shape

        # Statistics
        max_sinr_db = np.max(sinr_db_mat, axis=0)
        max_c = np.log2(1 + 10 ** (max_sinr_db / 10))
        max_c_mean = np.mean(max_c)

        sinr_t_lin = 10 ** (np.array(self.sinr_timeline) / 10)
        sinr_t_lin[np.isnan(sinr_t_lin)] = 0
        mean_c = np.mean(np.log2(1 + sinr_t_lin))

        q1_sinr, q3_sinr = np.quantile(sinr_db_mat, [0.25, 0.75])
        num_ho_exec_start = len(self.cntr["ho_exec"].start_idxs)
        rlf_rate = (
            len(self.cntr["rlfr"].start_idxs) / num_ho_exec_start
            if num_ho_exec_start > 0
            else 0
        )
        pp_rate = (
            len(self.cntr["mtsc"].aborted_idxs) / num_ho_exec_start
            if num_ho_exec_start > 0
            else 0
        )

        return {
            "num_base_stations": n_base_stations,
            "num_samples": n_samples,
            "mean_sinr": np.mean(sinr_db_mat),
            "variance_sinr": np.var(sinr_db_mat),
            "median_sinr": np.median(sinr_db_mat),
            "q1_sinr": q1_sinr,
            "q3_sinr": q3_sinr,
            "max_spectral_eff": max_c_mean,
            "spectral_eff": mean_c,
            "total_time": 0 + len(self.bs_idxs),
            "connected_time": 0
            + len(self.bs_idxs)
            - len(np.where((np.isnan(self.bs_idxs)))[0]),
            "num_ho_prep_started": 0 + len(self.cntr["ho_prep"].start_idxs),
            "num_ho_prep_terminated": 0 + len(self.cntr["ho_prep"].terminated_idxs),
            "num_ho_prep_completed": 0 + len(self.cntr["ho_prep"].done_idxs),
            "num_ho_prep_aborted": 0 + len(self.cntr["ho_prep"].aborted_idxs),
            "num_ho_prep_failed": 0 + len(self.cntr["ho_prep"].failed_idxs),
            "num_ho_prep_error": 0 + len(self.cntr["ho_prep"].error_idxs),
            "num_ho_exe_started": 0 + len(self.cntr["ho_exec"].start_idxs),
            "num_ho_exe_terminated": 0 + len(self.cntr["ho_exec"].terminated_idxs),
            "num_ho_exe_completed": 0 + len(self.cntr["ho_exec"].done_idxs),
            "num_ho_exe_aborted": 0 + len(self.cntr["ho_exec"].aborted_idxs),
            "num_ho_exe_failed": 0 + len(self.cntr["ho_exec"].failed_idxs),
            "num_ho_exe_error": 0 + len(self.cntr["ho_exec"].error_idxs),
            "num_rlf": 0 + len(self.cntr["rlfr"].start_idxs),
            "rlf_rate": 0.0 + rlf_rate,
            "num_pp": 0 + len(self.cntr["mtsc"].aborted_idxs),
            "pp_rate": 0.0 + pp_rate,
        }

    def get_stats_dict(self) -> dict[str, int | float | None]:
        """Return empty dict."""
        return {
            "num_base_stations": None,
            "num_samples": None,
            "mean_sinr": None,
            "mean_sinr_all_bs": None,
            "variance_sinr": None,
            "median_sinr": None,
            "q1_sinr": None,
            "q3_sinr": None,
            "max_spectral_eff": None,
            "spectral_eff": None,
            "total_time": None,
            "connected_time": None,
            "num_ho_prep_started": None,
            "num_ho_prep_terminated": None,
            "num_ho_prep_completed": None,
            "num_ho_prep_aborted": None,
            "num_ho_prep_failed": None,
            "num_ho_prep_error": None,
            "num_ho_exe_started": None,
            "num_ho_exe_terminated": None,
            "num_ho_exe_completed": None,
            "num_ho_exe_aborted": None,
            "num_ho_exe_failed": None,
            "num_ho_exe_error": None,
            "num_rlf": None,
            "rlf_rate": None,
            "num_pp": None,
            "pp_rate": None,
        }


def print_statistics(ho_3gpp_: HandoverEnv3GPP, sinr_db_mat: np.ndarray) -> None:
    """Print statistics."""
    n_base_stations, n_samples = sinr_db_mat.shape

    # Statistics
    max_sinr_db = np.max(sinr_db_mat, axis=0)
    max_c = np.log2(1 + 10 ** (max_sinr_db / 10))
    max_c_mean = np.mean(max_c)

    sinr_t_lin = 10 ** (np.array(ho_3gpp_.sinr_timeline) / 10)
    sinr_t_lin[np.isnan(sinr_t_lin)] = 0
    mean_c = np.mean(np.log2(1 + sinr_t_lin))

    q1_sinr, q3_sinr = np.quantile(sinr_db_mat, [0.25, 0.75])

    # Table with results
    params = [
        "Parameter",
        "Num BS",
        "Num samples",
        "",
        "Mean SINR",
        "Variance SINR",
        "Median SINR",
        "Q1 SINR",
        "Q3 SINR",
        "",
        "Max capacity",
        "Achieved capacity",
        "",
        "# TTT started",
        "# TTT finished",
        "# TTT abort",
        "",
        "# HO prep started",
        "# HO prep terminated",
        "# HO prep completed",
        "# HO prep aborted",
        "# HO prep failed",
        "# HO prep error",
        "",
        "# HO exe started",
        "# HO exe terminated",
        "# HO exe completed",
        "# HO exe aborted",
        "# HO exe failed",
        "# HO exe error",
        "",
        "# RLF",
        "# RLF rate",
        "# PP",
        "# PP rate",
    ]
    rlf_rate = (
        len(ho_3gpp_.cntr["rlfr"].start_idxs)
        / len(ho_3gpp_.cntr["ho_exec"].start_idxs)
        * 100
    )
    pp_rate = (
        len(ho_3gpp_.cntr["mtsc"].aborted_idxs)
        / len(ho_3gpp_.cntr["ho_exec"].start_idxs)
        * 100
    )
    values = [
        "Value",
        f"{n_base_stations}",
        f"{n_samples}",
        "",
        f"{np.mean(sinr_db_mat):.3f} dB",
        f"{np.var(sinr_db_mat):.3f} dB^2",
        f"{np.median(sinr_db_mat):.3f} dB",
        f"{q1_sinr:.3f} dB",
        f"{q3_sinr:.3f} dB",
        "",
        f"{max_c_mean:.3f} bits/s/Hz",
        f"{mean_c:.3f} bits/s/Hz | {100*mean_c/max_c_mean:.2f}%",
        "",
        f"{len(ho_3gpp_.cntr['tttc'].start_idxs)}",
        f"{len(ho_3gpp_.cntr['tttc'].terminated_idxs)}",
        f"{len(ho_3gpp_.cntr['tttc'].aborted_idxs)}",
        "",
        f"{len(ho_3gpp_.cntr['ho_prep'].start_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_prep'].terminated_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_prep'].done_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_prep'].aborted_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_prep'].failed_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_prep'].error_idxs)}",
        "",
        f"{len(ho_3gpp_.cntr['ho_exec'].start_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_exec'].terminated_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_exec'].done_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_exec'].aborted_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_exec'].failed_idxs)}",
        f"{len(ho_3gpp_.cntr['ho_exec'].error_idxs)}",
        "",
        f"{len(ho_3gpp_.cntr['rlfr'].start_idxs)}",
        f"{rlf_rate:.3f}%",
        f"{len(ho_3gpp_.cntr['mtsc'].aborted_idxs)}",
        f"{pp_rate:.3f}%",
    ]
    tab = list(zip(params, values))
    max_param_len = max(len(row[0]) for row in tab)

    print("Results")
    for row in tab:
        print(f"{row[0]:<{max_param_len}}: {row[1]}")
