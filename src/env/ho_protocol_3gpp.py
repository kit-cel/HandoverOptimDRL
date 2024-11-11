"""3GPP protocol modules."""

from abc import ABC, abstractmethod
import numpy as np


class AbstractTask(ABC):
    """Abstract task."""

    def __init__(
        self, name: str = None, debug: bool = False, debug_lvl: int = None
    ) -> None:
        self.name = name
        self.debug = debug
        self.debug_lvl = 0 if debug_lvl is None else debug_lvl

    def debug_msg(self, msg: str, lvl: int = None, tic: int = None) -> None:
        """Print debug message."""
        if self.debug and lvl <= self.debug_lvl:
            prefix = f"{self.name}" if tic is None else f"Tic{tic:05d} {self.name}"
            print(f"{prefix}: {msg}")

    @abstractmethod
    def reset(self, tic: int) -> None:
        """Reset task."""
        self.debug_msg("Reset", 0, tic)


class SyncSignal(AbstractTask):
    """SyncSignal task."""

    def __init__(
        self,
        q_in_db: float,
        q_out_db: float,
        debug: bool = False,
        debug_lvl: int = None,
    ) -> None:
        super().__init__("SyncSignal", debug, debug_lvl)
        self.q_in_db = q_in_db
        self.q_out_db = q_out_db
        self.flag_out_of_sync = False

    def reset(self, tic: int) -> None:
        super().reset(tic)
        self.flag_out_of_sync = False

    def check_out_of_sync(self, sinr_db: np.ndarray, bs: int, tic: int) -> None:
        """Check if out-of-sync."""
        if sinr_db[bs] < self.q_out_db:
            self.flag_out_of_sync = True
        elif sinr_db[bs] > self.q_in_db:
            self.flag_out_of_sync = False
        self.debug_msg(
            f"{'Out-of-sync' if self.flag_out_of_sync else 'In-sync'}: SINR {sinr_db[bs]:.2f} dB",
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
        debug_lvl: int = None,
    ) -> None:
        super().__init__("RRCConnectionReestablishment", debug, debug_lvl)
        self.q_in_db = q_in_db
        self.q_out_db = q_out_db
        self.pcell = np.nan
        self.ncell = np.nan

    def reset(self, tic: int) -> None:
        super().reset(tic)
        self.pcell = np.nan
        self.ncell = np.nan

    def rrc_measurement(self, rsrp_dbm: np.ndarray, tic: int) -> None:
        """RRC measurement."""
        ncell = np.argmax(rsrp_dbm)
        if ncell != self.pcell:
            self.ncell = ncell
            self.debug_msg(f"Neighboring cell detected: PCI {ncell}", 3, tic)
        else:
            self.ncell = np.nan
            self.debug_msg("Connected with best cell.", 3, tic)

    def cell_search(self, sinr_db: np.ndarray, tic: int) -> None:
        """Search for suitable cell."""
        cell = np.argmax(sinr_db)
        if sinr_db[cell] > self.q_in_db:
            self.ncell = cell
            self.debug_msg(f"Suitable cell found: PCI {cell}", 3, tic)
        else:
            self.ncell = np.nan
            self.debug_msg("No suitable cell found", 3, tic)

    def reconnection(self, tic: int) -> None:
        """Reconnect to suitable cell."""
        if not np.isnan(self.ncell):
            self.pcell = self.ncell
            self.ncell = np.nan
            self.debug_msg(f"Reconnected to cell: PCI {self.pcell}", 1, tic)

    def disconnect(self, tic: int) -> None:
        """Disconnect from cell."""
        self.debug_msg(f"Disconnected from cell: PCI {self.pcell}", 1, tic)
        self.pcell = np.nan

    @property
    def cell_detected(self) -> bool:
        """Return True if suitable cell is detected."""
        return not np.isnan(self.ncell)

    @property
    def is_connected(self) -> bool:
        """Return True if connected to cell."""
        return not np.isnan(self.pcell)


class HOEventHandler(AbstractTask):
    """3GPP handover task."""

    target_cell: int

    def __init__(self, debug: bool = False, **kwargs: dict[str, float | int]) -> None:
        super().__init__(
            "HOEventHandler", debug=debug, debug_lvl=kwargs.get("debug_lvl", 0)
        )

        self.rsrp_hys = {
            "a1": kwargs.get("hys_a1", 0),
            "a2": kwargs.get("hys_a2", 0),
            "a3": kwargs.get("hys_a3", 0),
            "a4": kwargs.get("hys_a4", 0),
            "a5": kwargs.get("hys_a5", 0),
            "a6": kwargs.get("hys_a6", 0),
            "b1": kwargs.get("hys_b1", 0),
            "b2": kwargs.get("hys_b2", 0),
        }

        self.rsrp_off = {
            "a1": kwargs.get("off_a1", 0),
            "a2": kwargs.get("off_a2", 0),
            "a3": kwargs.get("off_a3", 0),
            "a4": kwargs.get("off_a4", 0),
            "a5": kwargs.get("off_a5", 0),
            "a6": kwargs.get("off_a6", 0),
            "b1": kwargs.get("off_b1", 0),
            "b2": kwargs.get("off_b2", 0),
        }

        self.conditions = {
            "a1": False,
            "a2": False,
            "a3": False,
            "a4": False,
            "a5": False,
            "a6": False,
            "b1": False,
            "b2": False,
        }

    def reset(self, tic: int) -> None:
        """Reset task."""
        super().reset(tic)
        for condition in self.conditions:
            self.conditions[condition] = False
        self.target_cell = np.nan

    def event_a3(
        self,
        rsrp: np.ndarray,
        pcell: float,
        ncell: float,
        tic: int = None,
        ofn: float = 0.0,
        ocn: float = 0.0,
        ofp: float = 0.0,
        ocp: float = 0.0,
    ) -> bool:
        """Event A3 -- neighbor cell becomes offset better than SpCell
        3GPP TS 38.331 version 15.3.0 Release 15, Sec. 5.5.4.4

        Parameters
        ----------
        msn : float
           measure. result of the neighboring cell
        msp : float
           measure. result of the SpCell
        ofn : float (zero if not defined)
            measure. object specific offset of the reference signal of the neighbor cell
        ocn : float (zero if not defined)
            cell specific offset of the neighbor cell
        ofp : float (zero if not defined)
            measure. object specific offset of the SpCell
        ocp : float (zero if not defined)
            cell specific offset of the SpCel

        Returns
        -------
        bool
            Event A3
        """
        if np.isnan(ncell):
            return False
        if (
            rsrp[ncell] + ofn + ocn - self.rsrp_hys["a3"]
            > rsrp[pcell] + ofp + ocp + self.rsrp_off["a3"]
        ):
            self.conditions["a3"] = True  # Entering condition A3-1
            self.debug_msg("A3 entering condition.", 3, tic)
        if (
            rsrp[ncell] + ofn + ocn + self.rsrp_hys["a3"]
            < rsrp[pcell] + ofp + ocp + self.rsrp_off["a3"]
        ):
            self.conditions["a3"] = False  # Leaving condition A3-2
            self.debug_msg("A3 leaving condition.", 3, tic)
        return self.conditions["a3"]


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
        name: str = None,
        debug: bool = False,
        debug_lvl: int = None,
    ) -> None:
        super().__init__(name="SyncSignal", debug=debug)
        self.max_val = max_val
        self.step_size = step_size
        self.max_steps = max_val // step_size
        self.name = name
        self.debug = debug
        self.debug_lvl = 0 if debug_lvl is None else debug_lvl

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

    def reset(self, tic: int = None) -> None:
        """Reset task."""
        if self.flag_is_reset:
            return
        super().reset(tic)
        self.cnt = 0
        self.flag_is_reset = True
        self.flag_is_active = False
        self.flag_reached_max = False
        self.reset_idxs.append(tic)

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
