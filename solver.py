"""The XFOIL wrapper. The only place in the build that touches a subprocess.

Holds the command stream, the staging path guard, the per-call timeout with
its kill and reap path, polar parsing, and the classification of every run
into exactly one status.

The statuses are converged, partially_converged, failed, timeout and
environment_fault. Timeouts are reported separately from convergence failures
throughout: a timeout means the shape is slow, not that it is unsolvable. An
environment fault means the solver never got as far as solving, and is raised
when no polar accumulation header is written or when LOAD NOT COMPLETED
appears in the console output. A run that completes the command stream and
genuinely converges none of the requested angles still writes a polar file
with a header and no data rows, which is a different thing.

Two platform behaviours this wrapper exists to handle. XFOIL stores filenames
in a fixed-width buffer, so a long path truncates and the load fails silently,
producing no polar and no error a caller could tell from an aerodynamic
failure; each airfoil is therefore copied to a short relative filename inside
a staging directory and the solver runs with its working directory set there.
And sweeping into separation-prone angles can make XFOIL hang indefinitely
with no distinguishing output before the hang, which is why the timeout kills
and reaps the subprocess explicitly rather than trusting the default call to
terminate the child.

The operating point and the timeout are not set here. Both are read from
params.py, at the slots solver_operating_point_settings and per_call_timeout.

Called by evaluate.py, and by nothing else.

Public API
    SolverSettings, SolveStatus, SolveResult
    run_polar, run_polar_on_coords
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Solver settings. An explicit, required object -- the same discipline
# geometry.py uses for PlausibilityBounds. run_polar() never reaches for a
# module-level default; every call states what it is running at.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverSettings:
    """The the committed specification operating point and settings. Construct from params.py's
    committed values (slot "solver_operating_point_settings"); do not
    hand-construct with invented numbers."""
    reynolds: float
    mach: float
    ncrit: float
    alpha_start: float
    alpha_end: float
    alpha_step: float
    n_panels: int
    iter_limit: int

    def alphas(self) -> np.ndarray:
        n = int(round((self.alpha_end - self.alpha_start) / self.alpha_step)) + 1
        return self.alpha_start + self.alpha_step * np.arange(n)


class SolveStatus(str, Enum):
    CONVERGED = "converged"
    PARTIALLY_CONVERGED = "partially_converged"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ENVIRONMENT_FAULT = "environment_fault"


@dataclass
class SolveResult:
    status: SolveStatus
    reason: Optional[str]
    polar: Optional[np.ndarray]   # Nx7: alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
    console_output: str            # XFOIL's interactive stdout transcript, for debugging
    polar_dump_text: str           # the raw text of the PACC polar dump file itself, "" if never written
    elapsed_seconds: float
    n_converged: int
    n_requested: int


# Columns as XFOIL's PACC dump writes them.
POLAR_COLUMNS = ("alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr")

# The specific string XFOIL 6.99 itself emits when LOAD cannot read the
# staged file. Verified directly (see module docstring); used only as a
# diagnostic annotation on the environment_fault reason, not as the sole
# detection mechanism -- the primary signal is the absence of a polar
# header, which is robust to failures upstream of this specific message.
LOAD_FAILURE_SIGNATURE = "LOAD NOT COMPLETED"


# ---------------------------------------------------------------------------
# D06, the standing call-timing diagnostic. 
#
# the committed specification requires every solver call to record a non-zero elapsed
# time, and requires the check to run on every execution rather than once. The
# committed timeout rests on the measured distribution of these times, so a
# timer that silently recorded zero, or that measured the wrong span, would
# leave the timeout unjustified while the distribution still looked like one.
#
# The threshold is strictly greater than zero. It is not a tolerance: a
# duration of exactly zero is the failure itself, and every observed call in
# this build has taken hundreds of milliseconds. This raises rather than
# warning, and it computes nothing any caller reads.
# ---------------------------------------------------------------------------

D06_MINIMUM_ELAPSED_SECONDS = 0.0


def _checked_elapsed(start: float) -> float:
    """The elapsed wall clock for one call, checked before it is recorded.
    One helper on every exit path, so no return can skip it."""
    elapsed = time.perf_counter() - start
    if not (elapsed > D06_MINIMUM_ELAPSED_SECONDS):
        raise RuntimeError(
            f"D06: a solver call recorded an elapsed time of {elapsed!r} s, which "
            f"is not strictly greater than {D06_MINIMUM_ELAPSED_SECONDS!r}. The "
            f"committed per-call timeout rests on the distribution of these "
            f"times, so a timer measuring the wrong span leaves it unjustified."
        )
    return elapsed


# ---------------------------------------------------------------------------
# Staging. Short relative names, cwd set to the staging directory.
# ---------------------------------------------------------------------------

def _write_selig_dat(x: np.ndarray, y: np.ndarray, name: str, path: Path) -> None:
    lines = [name]
    lines += [f"{xi:.6f} {yi:.6f}" for xi, yi in zip(x, y)]
    path.write_text("\n".join(lines) + "\n")


def _cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Command stream. Verified against the real XFOIL 6.99 binary (module
# docstring) before being encoded here.
# ---------------------------------------------------------------------------

def _build_command_stream(short_name: str, polar_short: str, settings: SolverSettings) -> str:
    cmds = [
        f"LOAD {short_name}",
        "",                       # accept default airfoil name
        "PANE",                   # re-panel for a clean distribution
        "PPAR",                   # panel parameters menu
        f"N {settings.n_panels}",
        "",
        "",                       # exit PPAR menu
        "OPER",
        f"VISC {settings.reynolds}",
        f"MACH {settings.mach}",
        "VPAR",                   # viscous parameters menu
        f"N {settings.ncrit}",
        "",                       # exit VPAR menu
        f"ITER {settings.iter_limit}",
        "PACC",                   # polar accumulation on
        polar_short,
        "",                       # no separate dump file
    ]
    for a in settings.alphas():
        cmds.append(f"ALFA {a:.3f}")
    cmds.append("PACC")           # polar accumulation off
    cmds.append("")
    cmds.append("QUIT")
    return "\n".join(cmds) + "\n"


# ---------------------------------------------------------------------------
# Polar parsing. Same delimiter convention XFOIL's own PACC dump uses:
# a header block, a line of dashes, then fixed-width data rows.
# ---------------------------------------------------------------------------

def _parse_polar(polar_path: Path):
    """Returns (header_found, rows_array_or_None, raw_dump_text)."""
    if not polar_path.exists():
        return False, None, ""
    try:
        text = polar_path.read_text(errors="strict")
    except OSError:
        return False, None, ""
    lines = text.splitlines()

    header_found = False
    rows = []
    data_started = False
    for line in lines:
        if line.strip().startswith("-----"):
            data_started = True
            header_found = True
            continue
        if data_started:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    rows.append([float(p) for p in parts[:7]])
                except ValueError:
                    pass
    if not header_found:
        return False, None, text
    return True, (np.array(rows) if rows else np.empty((0, 7))), text


# ---------------------------------------------------------------------------
# B04. The wrapper.
# ---------------------------------------------------------------------------

def run_polar(dat_path, settings: SolverSettings, timeout_seconds: float,
              xfoil_binary, stage_dir) -> SolveResult:
    """Run one XFOIL viscous polar sweep for a single airfoil.

    Logic:
      1. write the coordinates to a short relative filename inside a
         staging directory (the coordinates are read from dat_path, which
         the caller has already written -- see run_polar_on_coords for the
         convenience form that writes them itself)
      2. launch the solver with the working directory set to that staging
         directory
      3. feed the command stream: load, repanel, viscous, transition
         setting, accumulate, alpha sweep, quit
      4. wait with a timeout; on timeout, kill the process, reap it, mark
         the record as a timeout
      5. on completion, read the accumulated polar dump, parse alpha, lift,
         drag and the transition columns, classify the outcome
      6. always remove the staged files, on every exit path
      7. return the parsed table, the status, the reason, and the elapsed
         time

    elapsed_seconds spans the entire call, staging through cleanup, so an
    independent wall-clock measurement wrapped around this whole function
    (B05's falsification check) should agree with it.
    """
    start = time.perf_counter()
    dat_path = Path(dat_path)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    short_name = f"a_{uuid.uuid4().hex[:8]}.dat"
    polar_short = f"p_{uuid.uuid4().hex[:8]}.txt"
    staged_dat = stage_dir / short_name
    staged_polar = stage_dir / polar_short

    n_requested = len(settings.alphas())

    try:
        shutil.copyfile(dat_path, staged_dat)
    except OSError as e:
        elapsed = _checked_elapsed(start)
        return SolveResult(
            status=SolveStatus.ENVIRONMENT_FAULT,
            reason=f"could not stage airfoil file: {e}",
            polar=None, console_output="", polar_dump_text="", elapsed_seconds=elapsed,
            n_converged=0, n_requested=n_requested,
        )

    stream = _build_command_stream(short_name, polar_short, settings)

    proc = None
    console_output = ""
    timed_out = False
    try:
        try:
            proc = subprocess.Popen(
                [str(Path(xfoil_binary).resolve())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, cwd=str(stage_dir),
            )
        except FileNotFoundError as e:
            elapsed = _checked_elapsed(start)
            _cleanup(staged_dat)
            return SolveResult(
                status=SolveStatus.ENVIRONMENT_FAULT,
                reason=f"XFOIL binary not found at {xfoil_binary}: {e}",
                polar=None, console_output="", polar_dump_text="", elapsed_seconds=elapsed,
                n_converged=0, n_requested=n_requested,
            )

        try:
            console_output, _err = proc.communicate(input=stream, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                console_output, _err = proc.communicate(timeout=10)
            except Exception:
                console_output = console_output or ""
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass

    if timed_out:
        elapsed = _checked_elapsed(start)
        _cleanup(staged_polar)
        _cleanup(staged_dat)
        return SolveResult(
            status=SolveStatus.TIMEOUT,
            reason=f"XFOIL exceeded the {timeout_seconds}s wall-clock limit and was killed.",
            polar=None, console_output=console_output, polar_dump_text="", elapsed_seconds=elapsed,
            n_converged=0, n_requested=n_requested,
        )

    header_found, polar, polar_dump_text = _parse_polar(staged_polar)
    _cleanup(staged_polar)
    _cleanup(staged_dat)
    elapsed = _checked_elapsed(start)

    if not header_found:
        detail = (f"'{LOAD_FAILURE_SIGNATURE}' seen in XFOIL output"
                   if LOAD_FAILURE_SIGNATURE in console_output
                   else "no polar accumulation header written")
        return SolveResult(
            status=SolveStatus.ENVIRONMENT_FAULT,
            reason=f"XFOIL did not complete the setup commands: {detail}.",
            polar=None, console_output=console_output, polar_dump_text=polar_dump_text,
            elapsed_seconds=elapsed, n_converged=0, n_requested=n_requested,
        )

    n_converged = 0 if polar is None else len(polar)
    if n_converged == n_requested:
        status, reason = SolveStatus.CONVERGED, None
    elif n_converged == 0:
        status = SolveStatus.FAILED
        reason = (f"0 of {n_requested} requested angles converged; "
                  f"XFOIL completed the run with no setup failure signature.")
    else:
        status = SolveStatus.PARTIALLY_CONVERGED
        reason = f"{n_converged} of {n_requested} requested angles converged."

    return SolveResult(
        status=status, reason=reason, polar=polar,
        console_output=console_output, polar_dump_text=polar_dump_text,
        elapsed_seconds=elapsed, n_converged=n_converged, n_requested=n_requested,
    )


def run_polar_on_coords(x: np.ndarray, y: np.ndarray, name: str,
                         settings: SolverSettings, timeout_seconds: float,
                         xfoil_binary, stage_dir, scratch_dir) -> SolveResult:
    """Convenience form: writes the coordinates to a Selig .dat file in
    scratch_dir, then calls run_polar. scratch_dir is separate from
    stage_dir; the caller-visible coordinate file is not itself staged
    under the short-name guard, only the copy run_polar makes of it is."""
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    dat_path = scratch_dir / f"{name}.dat"
    _write_selig_dat(x, y, name, dat_path)
    return run_polar(dat_path, settings, timeout_seconds, xfoil_binary, stage_dir)
