"""Radio Link Failure Discovery State Machine."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ho_optim_drl.config import Config


class AbstractTask(ABC):
    """Abstract task."""

    def __init__(
        self, name: str | None = None, debug: bool = False, verbose: int | None = None
    ) -> None:
        self.name = name
        self.debug = debug
        self.verbose = 0 if verbose is None else verbose

    def debug_msg(
        self, msg: str, lvl: int | None = None, tic: int | None = None
    ) -> None:
        """Print debug message."""
        if self.debug and lvl is not None and lvl <= self.verbose:
            prefix = f"{self.name}" if tic is None else f"Tic{tic:05d} {self.name}"
            print(f"{prefix}: {msg}")

    @abstractmethod
    def reset(self, tic: int | None) -> None:
        """Reset task."""
        self.debug_msg("Reset", 0, tic)


class SyncSignal(AbstractTask):
    """SyncSignal task."""

    def __init__(
        self,
        q_in_db: float,
        q_out_db: float,
        debug: bool = False,
        verbose: int | None = None,
    ) -> None:
        super().__init__("SyncSignal", debug, verbose)
        self.q_in_db = q_in_db
        self.q_out_db = q_out_db
        self.flag_out_of_sync = False

    def reset(self, tic: int | None) -> None:
        super().reset(tic)
        self.flag_out_of_sync = False

    def check_out_of_sync(self, sinr_db: np.ndarray, bs: int | None, tic: int) -> bool:
        """Check if out-of-sync."""
        if bs is None or bs < 0 or bs >= len(sinr_db):
            raise ValueError(f"Invalid base station index: {bs}")
        if sinr_db[bs] < self.q_out_db:
            self.flag_out_of_sync = True
        elif sinr_db[bs] > self.q_in_db:
            self.flag_out_of_sync = False
        self.debug_msg(
            f"{'Out-of-sync' if self.flag_out_of_sync else 'In-sync'}: "
            + f"SINR {sinr_db[bs]:.2f} dB, PCI {bs}",
            3,
            tic,
        )
        return self.flag_out_of_sync


class RadioResourceControl(AbstractTask):
    """RRCConnectionReestablishment task."""

    def __init__(
        self,
        q_in_db: float,
        q_out_db: float,
        debug: bool = False,
        verbose: int | None = None,
    ) -> None:
        super().__init__("RRCConnectionReestablishment", debug, verbose)
        self.q_in_db = q_in_db
        self.q_out_db = q_out_db
        self.pcell = None
        self.ncell = None
        self.tcell = None

    def reset(self, tic: int | None) -> None:
        super().reset(tic)
        self.pcell = None
        self.ncell = None
        self.tcell = None

    def update_ncell(self, ncell: int | None, tic: int) -> None:
        """RRC measurement."""
        if ncell is None:
            self.ncell = None
            self.debug_msg("Updated neighboring cell: None", 3, tic)
        else:
            if ncell != self.pcell:
                self.ncell = ncell
                self.debug_msg(f"Updated neighboring cell: PCI {ncell}", 3, tic)
            else:
                raise ValueError("Update of nCell failed, ncell is the same as pCell.")

    def update_tcell(self, tcell: int | None) -> None:
        """Update target cell."""
        self.tcell = tcell

    def cell_search(self, sinr_db: np.ndarray, tic: int) -> int | None:
        """Search for suitable cell."""
        cell = np.argmax(sinr_db).item()
        if sinr_db[cell] > self.q_in_db:
            self.ncell = cell
            self.debug_msg(f"Suitable cell found: PCI {cell}", 3, tic)
        else:
            self.ncell = None
            self.debug_msg("No suitable cell found", 3, tic)
        return cell

    def get_initial_cell(self, rsrp_db: np.ndarray) -> int:
        """Get initial cell."""
        self.pcell = np.argmax(rsrp_db).item()
        return self.pcell

    def reconnect(self, target_cell: int | None, tic: int) -> None:
        """Reconnect to suitable cell."""
        if target_cell is None:
            self.debug_msg("No target cell provided for reconnection", 1, tic)
            return
        self.pcell = target_cell
        self.ncell = None
        self.debug_msg(f"Reconnected to cell: PCI {self.pcell}", 1, tic)

    def disconnect(self, tic: int) -> None:
        """Disconnect from cell."""
        self.debug_msg(f"Disconnected from cell: PCI {self.pcell}", 1, tic)
        self.pcell = None

    @property
    def cell_detected(self) -> bool:
        """Return True if suitable cell is detected."""
        if self.ncell is None:
            return False
        return True

    @property
    def is_connected(self) -> bool:
        """Return True if connected to cell."""
        if self.pcell is None:
            return False
        return True


class GeneralCounter(AbstractTask):
    """Abstract task."""

    cnt: int
    flag_is_reset: bool
    flag_is_active: bool
    flag_reached_max: bool

    def __init__(
        self,
        max_val: int,
        step_size: int = 1,
        name: str | None = None,
        debug: bool = False,
        verbose: int | None = None,
    ) -> None:
        super().__init__(name="SyncSignal", debug=debug, verbose=verbose)
        self.max_val = max_val
        self.step_size = step_size
        self.max_steps = max_val // step_size
        self.name = name
        self.debug = debug
        self.verbose = 0 if verbose is None else verbose

        self.cnt = 0
        self.flag_is_reset = True
        self.flag_is_active = False
        self.flag_reached_max = False

        self.start_idxs = []  # Time step when counter started
        self.terminated_idxs = []  # Time step when counter reached max
        self.done_idxs = []  # Time step when task was done
        self.reset_idxs = []  # Time step when counter was reset

    def start(self, tic: int) -> None:
        """Start task."""
        if self.flag_is_reset:
            self.debug_msg(f"Start: {self.cnt}/{self.max_steps}", 0, tic)
            self.flag_is_active = True
            self.flag_is_reset = False
            self.start_idxs.append(tic)

    def step(self, tic: int) -> None:
        """Step task."""
        if self.flag_is_active:
            self.cnt += 1
            self.debug_msg(f"Step {self.cnt}/{self.max_steps}", 3, tic)
            if self.cnt >= self.max_steps:
                self.flag_is_active = False
                self.flag_reached_max = True
                self.terminated_idxs.append(tic)
                self.debug_msg(f"Max reached: {self.cnt}/{self.max_steps}", 1, tic)

    def reset(self, tic: int | None = None, reset_all: bool = False) -> None:
        """Reset task."""
        if self.flag_is_reset and not reset_all:
            return
        super().reset(tic)
        self.cnt = 0
        self.flag_is_reset = True
        self.flag_is_active = False
        self.flag_reached_max = False
        self.reset_idxs.append(tic)

        if reset_all:
            self.start_idxs = []
            self.terminated_idxs = []
            self.done_idxs = []
            self.reset_idxs = []
            print("Reset ", self.name, " pending", self.pending)

    def set_done(self, tic: int) -> None:
        """Set task as done."""
        self.done_idxs.append(tic)

    @property
    def active(self) -> bool:
        """Return True if task is active."""
        return self.flag_is_active

    @property
    def reached_max(self) -> bool:
        """Return True if counter reached max value."""
        return self.flag_reached_max

    @property
    def pending(self) -> bool:
        """Return True if task is pending."""
        return self.active or self.reached_max

    @property
    def aborted_idxs(self) -> list[int]:
        """Number of tasks aborted (before counter reached max)."""
        return sorted(set(self.reset_idxs) - set(self.terminated_idxs))

    @property
    def failed_idxs(self) -> list[int]:
        """Number of failed tasks (after counter terminated)."""
        return sorted(set(self.terminated_idxs) - set(self.done_idxs))

    @property
    def error_idxs(self) -> list[int]:
        """Number of error tasks (aborted and failed)."""
        return sorted(set(self.aborted_idxs) | set(self.failed_idxs))


class HOProcedurePPO:
    """3GPP handover task adapted for neural network predictor."""

    def __init__(self, config: "Config") -> None:
        """Initialize handover protocol."""
        self.config = config
        self.debug = False
        self.verbose = 0

        self.permit_ho_prep_abort = config.permit_ho_prep_abort

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

        # Flags
        self.ho_completed = False
        self.rlf_detected = False
        self.pp_detected = False

        self.tic = 0  # Step counter

    def reset(self) -> None:
        """Reset handover simulation."""
        self.sync_source.reset(self.tic)
        self.rrc.reset(self.tic)
        for task in self.cntr.values():
            task.reset(self.tic, reset_all=True)

        self._reset_flags()

        print(self.rrc.pcell)

        self.bs_idxs = []
        self.rsrp_timeline = []
        self.sinr_timeline = []
        self.sinr_at_ho_exe_pcell = []
        self.sinr_after_ho_exe_tcell = []

        self.max_mean_c = 0.0
        self.tic = 0

    def _reset_flags(self) -> None:
        self.ho_completed = False
        self.rlf_detected = False
        self.pp_detected = False

    def _step_counter_tasks(self, tic: int) -> None:
        """Step all counter tasks."""
        for task in self.cntr.values():  # Step all counter tasks
            task.step(tic)  # Do one step (if task is active)

    def _radio_link_monitoring(self, sinr_db: np.ndarray, tic: int) -> None:
        """Radio link monitoring (RLM)."""
        if self.rrc.is_connected:  # Connected to cell
            # Out-of-sync signal detected
            if self.sync_source.check_out_of_sync(sinr_db, self.rrc.pcell, tic):
                # Out-of-sync signal, reset N311 counter and start N310 counter
                self.cntr["n311"].reset(tic)
                if not self.cntr["t310"].pending:
                    # T310 not running, start N310 counter
                    self.cntr["n310"].start(tic)
                    # print("Out-of-sync signal detected")
            else:  # In-sync signal detected
                # Reset N310 counter
                self.cntr["n310"].reset(tic)
                if self.cntr["t310"].pending:  # In-sync signal and T310 running
                    # Start N311 counter
                    self.cntr["n311"].start(tic)
        else:  # Disconnected from cell, reset N310, N311, T310
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)

    def _pp_monitoring(self, tic: int, start: bool = False) -> None:
        """
        Ping-pong monitoring.

        Number of PPs observed:
            self.mtsc.aborted_idxs
        """
        if start:  # Start ping-pong monitoring
            if self.cntr["mtsc"].pending:
                self.pp_detected = True
            self.cntr["mtsc"].reset(tic)
            self.cntr["mtsc"].start(tic)
        else:  # PP monitoring ongoing
            if self.cntr["mtsc"].reached_max:
                # MTS expired, reset PP monitoring
                self.cntr["mtsc"].reset(tic)

    def _rlf_detection(self, tic: int) -> int:
        """Detection of radio link failure (RLF)."""
        # N310 max reached, start timer T310
        if self.cntr["n310"].reached_max:
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].start(tic)
        # N311 max reached, reset timer T310
        elif self.cntr["n311"].reached_max:
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)
        # T310 expired, declare RLF
        elif self.cntr["t310"].reached_max:
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)
            return 1  # RLF detected
        return 0  # No RLF detected

    def _rlf_recovery(self, tic: int, start: bool = False) -> None:
        """RLF recovery."""
        if start:  # Start RLF recovery
            # Start RLF recovery timer
            self.cntr["rlfr"].start(tic)
            # Disconnect from cell
            self.rrc.disconnect(tic)
            self.rlf_detected = True
            # print("RLF detected")
        elif self.cntr["rlfr"].reached_max:  # RLF recovery timer expired
            # Suitable cell detected, reconnect
            if self.rrc.cell_detected:
                # Get target cell
                ncell = self.rrc.ncell
                # Reconnect to target cell
                self.rrc.reconnect(ncell, tic)
                # Reset RLF recovery counter
                self.cntr["rlfr"].reset(tic)
            else:  # No suitable cell detected, continue RLF recovery
                pass
        else:  # RLF recovery ongoing
            pass

    def _ho_preparation_handler(self, tic: int) -> None:
        """Event handler."""
        if self.rrc.is_connected:  # Connected to cell
            pcell = self.rrc.pcell
            tcell = self.rrc.tcell

            # Trigger HO preparation if target cell is not None and different from current cell
            if tcell is not None and tcell != pcell:

                # Allow HO aborting
                if self.permit_ho_prep_abort:
                    # Reset HO preparation due to target cell change
                    if tcell != self.rrc.ncell and self.cntr["ho_prep"].pending:
                        self.cntr["ho_prep"].reset(tic)

                # (Re-)Start HO preparation if not already ongoing
                if not self.cntr["ho_prep"].pending:
                    self.rrc.update_ncell(tcell, tic)
                    self.cntr["ho_prep"].start(tic)

            # Abort HO preparation and stay in current cell (if allowed)
            elif self.permit_ho_prep_abort and self.cntr["ho_prep"].pending:
                # Cancel HO preparation
                self.cntr["ho_prep"].reset(tic)
                # Update RRC nCell
                self.rrc.update_ncell(None, tic)
            else:
                pass

        else:  # Disconnected from cell, reset TTT and HO preparation
            self.cntr["ho_prep"].reset(tic)

    def _ho_state_machine(self, sinr_db: np.ndarray, tic: int) -> None:
        """Handover state machine."""
        if self.cntr["ho_prep"].reached_max:
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
            tcell = self.rrc.ncell
            self.cntr["ho_exec"].reset(tic)

            self.sinr_after_ho_exe_tcell.append(sinr_db[tcell])

            # Check if target cell is reachable
            if self.sync_source.check_out_of_sync(sinr_db, tcell, tic):
                # Target BS not reachable, declare RLF and start RLF recovery
                self._rlf_recovery(tic, start=True)
            else:
                # Connect to target cell
                self.cntr["ho_exec"].set_done(tic)
                self.rrc.reconnect(tcell, tic)
                self.ho_completed = True
                # print("Handover completed")

    def _add_results_to_timeline(
        self, rsrp_i_db: np.ndarray, sinr_i_db: np.ndarray
    ) -> None:
        """Add results to timeline."""
        if self.rrc.is_connected:
            self.bs_idxs.append(self.rrc.pcell)
            self.rsrp_timeline.append(rsrp_i_db[self.rrc.pcell])
            self.sinr_timeline.append(sinr_i_db[self.rrc.pcell])
        else:
            self.bs_idxs.append(np.nan)
            self.rsrp_timeline.append(np.nan)
            self.sinr_timeline.append(np.nan)

    def _mean_capacity(self) -> float:
        """Mean capacity."""
        sinr_t_lin = 10 ** (np.array(self.sinr_timeline) / 10)
        sinr_t_lin[np.isnan(sinr_t_lin)] = 0
        return np.mean(np.log2(1 + sinr_t_lin)).item() if len(sinr_t_lin) > 0 else 0.0

    def step(
        self,
        rsrp_db: np.ndarray,
        sinr_db: np.ndarray,
        target_cell: int | None = None,
    ) -> dict[str, int | float]:
        """Validate a general handover protocol."""
        tic = self.tic

        self._reset_flags()

        if tic == 0:
            # Max capacity (upper bound)
            self.max_mean_c = np.mean(np.log2(1 + 10 ** (np.max(sinr_db, axis=0) / 10)))
            # Get initial cell
            pcell = self.rrc.get_initial_cell(rsrp_db)
            # Initial connection
            self.rrc.reconnect(pcell, tic)

        # Update target cell
        self.rrc.update_tcell(target_cell)

        # Timer and counter tasks
        self._step_counter_tasks(tic)

        # Monitoring
        self._radio_link_monitoring(sinr_db, tic)  # Radio link monitoring

        # Ping-pong monitoring
        self._pp_monitoring(tic)

        # RLF detection (N310, N311, T310)
        if self._rlf_detection(tic):
            # RLF detected, start RLF recovery
            self._rlf_recovery(tic, start=True)

        # Measurement report event handler
        self._ho_preparation_handler(tic)

        # Handover state machine (HO preparation, HO execution timers)
        self._ho_state_machine(sinr_db, tic)

        # RLF recovery
        self._rlf_recovery(tic)

        # Add results to timeline
        self._add_results_to_timeline(rsrp_db, sinr_db)

        # Next time step
        self.tic += 1

        # Return state dict
        return self._get_state_dict()

    def get_pcell(self) -> int | None:
        """Get primary cell."""
        return self.rrc.pcell

    def get_tcell(self) -> int | None:
        """Get target cell."""
        return self.rrc.ncell

    def _get_state_dict(self) -> dict[str, int | float]:
        """Get state dictionary."""
        return {
            "q_in_out": self.sync_source.flag_out_of_sync,
            "ho_prep": self.cntr["ho_prep"].pending,
            "ho_exec": self.cntr["ho_exec"].pending,
            "rlf": self.rlf_detected,
            "pp": self.pp_detected,
            "n310_t310_rel_cnt": (self.cntr["n310"].cnt + self.cntr["t310"].cnt)
            / (self.cntr["n310"].max_val + self.cntr["t310"].max_val),
            "ho_prep_rel_cnt": self.cntr["ho_prep"].cnt / self.cntr["ho_prep"].max_val,
            "ho_exec_rel_cnt": self.cntr["ho_exec"].cnt / self.cntr["ho_exec"].max_val,
            "mtsc_rel_cnt": self.cntr["mtsc"].cnt / self.cntr["mtsc"].max_val,
            "ho_complete": self.ho_completed,
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

    def get_statistics(self, sinr_db_mat: np.ndarray) -> dict[str, int | float | None]:
        """Get statistics."""
        sinr_db_mat = sinr_db_mat.T
        sinr_lin_mat = 10 ** (sinr_db_mat / 10)
        n_base_stations, n_samples = sinr_db_mat.shape

        # Statistics
        max_sinr_db = np.max(sinr_db_mat, axis=0)
        max_c = np.log2(1 + 10 ** (max_sinr_db / 10))
        max_c_mean = np.mean(max_c).item()

        sinr_t_lin = 10 ** (np.array(self.sinr_timeline) / 10)
        sinr_t_lin[np.isnan(sinr_t_lin)] = 1e-10
        mean_sinr_db = 10 * np.log10(np.mean(sinr_t_lin))
        var_sinr_db = 10 * np.log10(np.var(sinr_t_lin))
        median_sinr_db = 10 * np.log10(np.median(sinr_t_lin))
        q1_sinr, q3_sinr = np.quantile(sinr_t_lin, [0.25, 0.75])

        mean_sinr_db_all_bs = 10 * np.log10(np.mean(sinr_lin_mat))

        mean_c = np.mean(np.log2(1 + sinr_t_lin)).item()

        q1_sinr, q3_sinr = np.quantile(sinr_db_mat, [0.25, 0.75])
        num_ho_exec_start = len(self.cntr["ho_exec"].start_idxs)
        hof_rate = (
            len(self.cntr["rlfr"].start_idxs) / num_ho_exec_start
            if num_ho_exec_start > 0
            else 0.0
        )

        pp_rate = (
            len(self.cntr["mtsc"].aborted_idxs) / num_ho_exec_start
            if num_ho_exec_start > 0
            else 0.0
        )

        stats = self.get_stats_dict()
        stats.update(
            {
                "num_base_stations": n_base_stations,
                "num_samples": n_samples,
                "mean_sinr": float(mean_sinr_db),
                "mean_sinr_all_bs": float(mean_sinr_db_all_bs),
                "variance_sinr": float(var_sinr_db),
                "median_sinr": float(median_sinr_db),
                "q1_sinr": float(q1_sinr),
                "q3_sinr": float(q3_sinr),
                "max_spectral_eff": float(max_c_mean),
                "spectral_eff": float(mean_c),
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
                "rlf_rate": 0.0 + float(hof_rate),
                "num_pp": 0 + len(self.cntr["mtsc"].aborted_idxs),
                "pp_rate": 0.0 + float(pp_rate),
            }
        )

        return stats


def print_statistics(stat_dict: dict[str, int | float]) -> None:
    """Print statistics."""
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
        "Total time",
        "Connected time",
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
    values = [
        "Value",
        f"{stat_dict['num_base_stations']}",
        f"{stat_dict['num_samples']}",
        "",
        f"{stat_dict['mean_sinr']:.3f} dB",
        f"{stat_dict['variance_sinr']:.3f} dB^2",
        f"{stat_dict['median_sinr']:.3f} dB",
        f"{stat_dict['q1_sinr']:.3f} dB",
        f"{stat_dict['q3_sinr']:.3f} dB",
        "",
        f"{stat_dict['max_spectral_eff']:.3f} bits/s/Hz",
        f"{stat_dict['spectral_eff']:.3f} bits/s/Hz |"
        + f" {100 * stat_dict['spectral_eff'] / stat_dict['max_spectral_eff']:.3f}%",
        f"{stat_dict['total_time']} samples",
        f"{stat_dict['connected_time']} samples |"
        + f" {100 * stat_dict['connected_time'] / stat_dict['total_time']:.3f}%",
        "",
        f"{stat_dict['num_ho_prep_started']}",
        f"{stat_dict['num_ho_prep_terminated']}",
        f"{stat_dict['num_ho_prep_completed']}",
        f"{stat_dict['num_ho_prep_aborted']}",
        f"{stat_dict['num_ho_prep_failed']}",
        f"{stat_dict['num_ho_prep_error']}",
        "",
        f"{stat_dict['num_ho_exe_started']}",
        f"{stat_dict['num_ho_exe_terminated']}",
        f"{stat_dict['num_ho_exe_completed']}",
        f"{stat_dict['num_ho_exe_aborted']}",
        f"{stat_dict['num_ho_exe_failed']}",
        f"{stat_dict['num_ho_exe_error']}",
        "",
        f"{stat_dict['num_rlf']}",
        f"{100 * stat_dict['rlf_rate']:.3f}%",
        f"{stat_dict['num_pp']}",
        f"{100 * stat_dict['pp_rate']:.3f}%",
    ]
    tab = list(zip(params, values))
    max_param_len = max(len(row[0]) for row in tab)

    print("Results")
    for row in tab:
        print(f"{row[0]:<{max_param_len}}: {row[1]}")
