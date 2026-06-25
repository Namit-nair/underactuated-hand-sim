"""CSV logger for the tendon-driven finger hardware validation rig.

Writes one row per capture into the SAME folder as the MuJoCo simulation CSVs
(``high_fidelity/validation_results/``) so hardware and simulation results live
together. Each logger instance owns a single timestamped file whose name
encodes the spring-set label, e.g.::

    hw_validation_<label>_<YYYYmmdd_HHMMSS>.csv

The column order is fixed (see :attr:`CsvLogger.COLUMNS`). :meth:`CsvLogger.log`
is tolerant: unknown keys are ignored, missing columns are blank-filled, and
floats are rounded to 6 decimal places without crashing on ``None``.
"""

import csv
import os
import re
from datetime import datetime

DEFAULT_OUT_DIR = "/home/namit/iitgn/underactuated_finger/high_fidelity/validation_results"

_FLOAT_PRECISION = 6


def _sanitize_label(label: str) -> str:
    """Make a label safe for use inside a filename."""
    label = str(label).strip()
    # Collapse anything that is not alphanumeric / dash / underscore into '_'.
    label = re.sub(r"[^0-9A-Za-z._-]+", "_", label)
    label = label.strip("_")
    return label or "custom"


def _format_value(value):
    """Format a single cell value for CSV output.

    ``None`` -> "" ; floats are rounded; everything else is passed through.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        # bool is a subclass of int — keep it readable and avoid 0/1 rounding.
        return value
    if isinstance(value, float):
        try:
            return round(value, _FLOAT_PRECISION)
        except (ValueError, OverflowError):
            return value
    return value


class CsvLogger:
    """Append-style CSV logger; one file per spring-set run."""

    COLUMNS = [
        "timestamp",
        "spring_set_label",
        "rho1",
        "rho3",
        "k_mcp",
        "k_pip",
        "k_dip",
        "delta_L_mm",
        "servo_pos",
        "servo_current",
        "theta_mcp_exp",
        "theta_pip_exp",
        "theta_dip_exp",
        "theta_mcp_ana",
        "theta_pip_ana",
        "theta_dip_ana",
        "err_mcp",
        "err_pip",
        "err_dip",
        "M12_exp",
        "M32_exp",
        "M12_ana",
        "M32_ana",
        "markers_all_visible",
        "settle_time_s",
        "trial_idx",
    ]

    def __init__(self, spring_set_label: str = "custom",
                 out_dir: str = DEFAULT_OUT_DIR,
                 columns: list = None,
                 filename_prefix: str = "hw_validation"):
        """One CSV file per run.

        Defaults reproduce the validation-rig behaviour (``hw_validation_*`` with
        :attr:`COLUMNS`). Pass ``columns`` and ``filename_prefix`` to reuse the
        same tolerant writer for another rig — e.g. the load-carrying pull-out
        test uses ``filename_prefix="hw_loadtest"`` with its own column set.
        """
        self.spring_set_label = spring_set_label
        self.out_dir = out_dir
        # Per-instance columns (falls back to the class default) so the tolerant
        # blank-fill / round / None-safe formatting is shared across rigs.
        self.columns = list(columns) if columns is not None else list(self.COLUMNS)
        os.makedirs(out_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = _sanitize_label(spring_set_label)
        fname = f"{filename_prefix}_{safe_label}_{stamp}.csv"
        self._filepath = os.path.join(out_dir, fname)

        self._n_rows = 0
        self._fh = open(self._filepath, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._fh.flush()

    @property
    def filepath(self) -> str:
        """Absolute path of the CSV file this logger writes to."""
        return self._filepath

    def log(self, row: dict) -> None:
        """Write one row from ``row`` (keyed by column name).

        Unknown keys are ignored, missing columns are blank-filled, floats are
        rounded, and ``None`` values become empty cells.
        """
        out = {col: _format_value(row.get(col)) for col in self.columns}
        self._writer.writerow(out)
        self._fh.flush()
        self._n_rows += 1

    def n_rows(self) -> int:
        """Number of data rows written so far (excludes the header)."""
        return self._n_rows

    def close(self) -> None:
        """Close the underlying file handle (idempotent)."""
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
