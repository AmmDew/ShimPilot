"""ShimPilot: observation-only core-starvation tripwire for "System Lagging".

selfdrived's SCHED_FIFO loop is pinned to core 5; when an unknown co-resident
saturates that core, the Ratekeeper trips and openpilot soft-disables with
"System Lagging". This module identifies the co-resident by keeping a bounded
ring of per-process CPU deltas sampled from /proc.

Safety properties (non-negotiable):
- The sampler thread pins itself to background cores 0-2 under SCHED_IDLE
  (nice 19 fallback), mirroring starport/placement.py: ordinary driving work
  always preempts it and it cannot compete on the controls/model cores 3-7.
- Nothing here runs inside the 10 ms control loop: the hot path may only
  call note_lag()/note_ok(), which set/clear a threading.Event.
- Pure /proc reads and dict math; no imports of messaging, params, or
  hardware stacks, so it cannot block on or perturb anything that drives.
- Observability only: no alert, threshold, affinity, or safety behavior in
  openpilot is modified by this module.

Typical host use (standalone daemon or RT process sidecar):

    from openpilot.starpilot import lag_watch
    watch = lag_watch.LagWatch()
    watch.start()          # spawns the SCHED_IDLE sampler thread
    ...
    watch.note_lag()       # rising edge of selfdrivedLagging (ns, event.set)
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

BACKGROUND_CORES = (0, 1, 2)   # never 3-7: driving cores, per starport placement policy
SAMPLE_SECONDS = 5.0
RING_ENTRIES = 36               # 3 minutes of history
FLUSH_MIN_INTERVAL = 60.0       # at most one log event per minute while lagging persists
TOP_N = 12
KERNEL_TOP_N = 3                # kernel workers can be the real culprit; keep them in view
PAYLOAD_BUDGET_BYTES = 6000     # keep each log event comfortably under capnp/swaglog comfort
FALLBACK_NICE = 19


def _demangle_cmdline(args: list[str], comm: str) -> tuple[str, bool]:
  """Return (display name, is_kernel). Mirrors galaxy/system_monitor.py naming so
  profiles line up with what Galaxy's system page shows."""
  if not args:
    return f"[{comm}]", True
  name = args[0]
  if "python" in Path(name).name and len(args) > 1:
    name = args[2] if args[1] == "-m" and len(args) > 2 else args[1] if not args[1].startswith("-") else Path(name).name
  return name.removeprefix("/data/openpilot/"), False


class ProcSampler:
  """Per-process CPU% deltas between successive sample() calls, plus per-core busy%."""

  def __init__(self, root: Path = Path("/proc"), clock=time.monotonic, tick_hz: int | None = None,
               ring_entries: int = RING_ENTRIES):
    self.root = Path(root)
    self.clock = clock
    self.tick_hz = tick_hz if tick_hz is not None else os.sysconf("SC_CLK_TCK")
    self.ring = deque(maxlen=ring_entries)
    self._prev_ticks: dict[tuple[int, int], int] = {}
    self._prev_cores: dict[str, tuple[int, int]] = {}   # label -> (total, busy) jiffies
    self._prev_time: float | None = None

  def _read_cores(self) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for line in (self.root / "stat").read_text().splitlines():
      label, _, rest = line.partition(" ")
      if not label.startswith("cpu"):
        continue
      values = [int(field) for field in rest.split()]
      idle = values[3] + (values[4] if len(values) > 4 else 0)   # idle + iowait
      out[label] = (sum(values), idle)
    return out

  def _read_procs(self) -> tuple[dict[tuple[int, int], tuple[int, str, bool]], int]:
    procs: dict[tuple[int, int], tuple[int, str, bool]] = {}
    try:
      names = os.listdir(self.root)
    except OSError:
      return procs, 0
    for entry in names:
      if not entry.isdigit():
        continue
      pid = int(entry)
      if pid <= 0:
        continue
      try:
        raw = (self.root / entry / "stat").read_text()
        end = raw.rindex(")")
        fields = raw[end + 2:].split()
        identity = (pid, int(fields[19]))
        ticks = int(fields[11]) + int(fields[12])
        args = [part for part in (self.root / entry / "cmdline").read_bytes().decode(errors="replace").split("\0") if part]
        procs[identity] = (ticks, *_demangle_cmdline(args, raw[raw.index("(") + 1:end]))
      except (OSError, ValueError, IndexError):
        continue   # processes vanish between listing and reading; skip the slice for them
    return procs, len(procs)

  def sample(self) -> None:
    now = self.clock()
    cores = self._read_cores()
    procs, count = self._read_procs()
    duration = 0.0 if self._prev_time is None else max(now - self._prev_time, 1e-6)
    core_pcts: dict[str, float] = {}
    for label, (total, idle) in cores.items():
      prev = self._prev_cores.get(label)
      if prev is not None and total > prev[0]:
        busy_delta = (total - prev[0]) - (idle - prev[1])
        core_pcts[label] = round(100.0 * busy_delta / (total - prev[0]), 1)
    pcts: list[dict] = []
    if self._prev_ticks and duration > 0:
      for identity, (ticks, name, kernel) in procs.items():
        prev_ticks = self._prev_ticks.get(identity)
        if prev_ticks is None or ticks < prev_ticks:
          continue   # new or restarted process: next slice will cover it
        cpu = (ticks - prev_ticks) / (self.tick_hz * duration) * 100.0
        if cpu >= 1.0:
          pcts.append({"n": name, "c": round(cpu, 1), "k": kernel})
    pcts.sort(key=lambda row: row["c"], reverse=True)
    self.ring.append({"t": round(now, 3), "dur": round(duration, 3), "procs": pcts, "cores": core_pcts, "count": count})
    self._prev_ticks = {identity: ticks for identity, (ticks, _n, _k) in procs.items()}
    self._prev_cores = cores
    self._prev_time = now

  def payload(self, extra: dict | None = None, budget: int = PAYLOAD_BUDGET_BYTES) -> dict:
    """Trimmed profile of the most recent slice: top userspace, top kernel, per-core busy%."""
    if not self.ring:
      return {"procs": [], "cores": {}, "empty": True}
    latest = self.ring[-1]
    userspace = [row for row in latest["procs"] if not row["k"]][:TOP_N]
    kernels = [row for row in latest["procs"] if row["k"]][:KERNEL_TOP_N]
    body = {"t": latest["t"], "dur": latest["dur"], "cores": latest["cores"]}
    if extra:
      body.update(extra)
    chosen: list[dict] = []
    for row in userspace + kernels:
      candidate = chosen + [row]
      probe = dict(body, procs=candidate)
      if len(json.dumps(probe).encode()) > budget and len(chosen) >= 4:
        break
      chosen = candidate
    return dict(body, procs=chosen, dropped=len(latest["procs"]) - len(chosen))


class LagWatch:
  """Background SCHED_IDLE sampler with edge-triggered flush. Host supplies on_flush(dict)."""

  def __init__(self, sampler: ProcSampler | None = None, on_flush=None,
               sample_seconds: float = SAMPLE_SECONDS, flush_min_interval: float = FLUSH_MIN_INTERVAL):
    self.sampler = sampler or ProcSampler()
    self.on_flush = on_flush or (lambda payload: None)
    self.sample_seconds = sample_seconds
    self.flush_min_interval = flush_min_interval
    self._lag_event = threading.Event()
    self._stop_event = threading.Event()
    self._last_flush: float | None = None
    self._thread: threading.Thread | None = None
    self._pump_count = 0

  # --- hot-path API: O(1), no I/O, no locks beyond an Event flag ---
  def note_lag(self) -> None:
    self._lag_event.set()

  def note_ok(self) -> None:
    self._lag_event.clear()

  # --- mechanics, kept separate from threading so tests drive it directly ---
  def pump(self, now: float | None = None) -> None:
    try:
      self.sampler.sample()
      self._pump_count += 1
      if not self._lag_event.is_set():
        self._last_flush = None   # re-arm: next rising edge flushes immediately
        return
      stamp = self.sampler.clock() if now is None else now
      if self._last_flush is None or stamp - self._last_flush >= self.flush_min_interval:
        self._last_flush = stamp
        self.on_flush(self.sampler.payload())
    except Exception:
      pass   # observation must never take the host down

  def start(self) -> "LagWatch":
    if self._thread is not None:
      return self
    self._thread = threading.Thread(target=self._run, name="lag_watch", daemon=True)
    self._thread.start()
    return self

  def stop(self) -> None:
    self._stop_event.set()

  def _run(self) -> None:
    try:   # background-core placement; never on 3-7, never above ordinary work
      os.sched_setaffinity(0, set(BACKGROUND_CORES) & set(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
      pass
    try:
      os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
      try:
        os.nice(FALLBACK_NICE)
      except OSError:
        pass
    while not self._stop_event.wait(self.sample_seconds):
      try:
        self.pump()
      except Exception:
        pass   # a broken sampler must never take the host down
