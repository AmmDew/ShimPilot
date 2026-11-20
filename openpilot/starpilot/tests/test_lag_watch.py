import json
from pathlib import Path

import pytest

from openpilot.starpilot.lag_watch import LagWatch, ProcSampler


class FakeClock:
  def __init__(self):
    self.now = 1000.0

  def __call__(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


def _proc_stat(pid, comm, utime, stime, starttime):
  # fields after ") ": state@0 ... utime@11, stime@12, starttime@19
  tokens = ["0"] * 20
  tokens[0] = "S"
  tokens[11] = str(utime)
  tokens[12] = str(stime)
  tokens[19] = str(starttime)
  return f"{pid} ({comm}) " + " ".join(tokens) + "\n"


def _write_proc(root: Path, pid, comm, utime=0, stime=0, starttime=500, cmdline=b""):
  d = root / str(pid)
  d.mkdir(parents=True, exist_ok=True)
  (d / "stat").write_text(_proc_stat(pid, comm, utime, stime, starttime))
  (d / "cmdline").write_bytes(cmdline)


def _write_proc_stat(root: Path, cores: dict[str, tuple[int, int]]):
  # cores: name -> (total, idle+iowait) is encoded back into raw fields: user+system... busy=total-idle
  lines = []
  for label, (total, idle) in cores.items():
    values = [0] * 10
    values[3] = idle                       # idle
    values[0] = total - idle               # user carries the busy remainder
    lines.append(label + " " + " ".join(map(str, values)))
  (root / "stat").write_text("\n".join(lines) + "\n")


@pytest.fixture
def fake_proc(tmp_path):
  root = tmp_path / "proc"
  root.mkdir()
  _write_proc_stat(root, {"cpu0": (1000, 900), "cpu5": (1000, 900)})
  _write_proc(root, 100, "selfdrived", utime=1000, cmdline=b"python\x00-m\x00openpilot.selfdrive.selfdrived\x00\x00")
  _write_proc(root, 101, "python", utime=100, cmdline=b"python\x00-m\x00openpilot.upload\x00\x00")
  _write_proc(root, 102, "kworker/u16:0", utime=0, cmdline=b"")     # kernel thread
  return root


def _bump(root: Path, pid, utime):
  stat = (root / str(pid) / "stat").read_text()
  head, _, tail = stat.rpartition(") ")
  fields = tail.split()
  fields[11] = str(utime)
  (root / str(pid) / "stat").write_text(head + ") " + " ".join(fields) + "\n")


def test_cpu_delta_and_naming(fake_proc):
  clock = FakeClock()
  sampler = ProcSampler(root=fake_proc, clock=clock, tick_hz=100)
  sampler.sample()                      # seeds baselines
  clock.advance(5.0)
  _write_proc_stat(fake_proc, {"cpu0": (1500, 1000), "cpu5": (1500, 1050)})
  _bump(fake_proc, 101, 100 + 500)      # uploader burns 500 ticks in 5 s = 100%
  _bump(fake_proc, 100, 1000 + 100)     # selfdrived 100 ticks = 20%
  sampler.sample()
  top = sampler.ring[-1]["procs"]
  assert top[0] == {"n": "openpilot.upload", "c": 100.0, "k": False}
  assert {"n": "openpilot.selfdrive.selfdrived", "c": 20.0, "k": False} in top
  assert sampler.ring[-1]["cores"]["cpu5"] == pytest.approx(70.0)
  assert sampler.ring[-1]["cores"]["cpu0"] == pytest.approx(80.0)


def test_kernel_thread_flagged(fake_proc):
  clock = FakeClock()
  sampler = ProcSampler(root=fake_proc, clock=clock, tick_hz=100)
  sampler.sample()
  clock.advance(5.0)
  _bump(fake_proc, 102, 400)            # kworker 400 ticks / 5 s = 80%
  sampler.sample()
  rows = {row["n"]: row for row in sampler.ring[-1]["procs"]}
  assert rows["[kworker/u16:0]"]["k"] is True


def test_restart_identity_not_negative(fake_proc):
  clock = FakeClock()
  sampler = ProcSampler(root=fake_proc, clock=clock, tick_hz=100)
  sampler.sample()
  clock.advance(5.0)
  _write_proc(fake_proc, 101, "python", utime=3, starttime=9999, cmdline=b"python\x00-m\x00openpilot.upload\x00\x00")  # restarted, ticks reset
  sampler.sample()
  assert all(row["c"] >= 0 for row in sampler.ring[-1]["procs"])
  clock.advance(5.0)
  _bump(fake_proc, 101, 503)
  sampler.sample()                      # new identity now measurable
  assert any(row["n"] == "openpilot.upload" and row["c"] >= 99 for row in sampler.ring[-1]["procs"])


def test_ring_eviction(fake_proc):
  clock = FakeClock()
  sampler = ProcSampler(root=fake_proc, clock=clock, tick_hz=100, ring_entries=4)
  for _ in range(10):
    clock.advance(5.0)
    sampler.sample()
  assert len(sampler.ring) == 4


def test_payload_budget_and_kernel_reservation(tmp_path):
  root = tmp_path / "proc"
  root.mkdir()
  _write_proc_stat(root, {"cpu0": (1000, 990)})
  for pid in range(500, 700):
    _write_proc(root, pid, "x" * 180, utime=0, cmdline=("python\x00" + "y" * 180 + "\x00\x00").encode())
  _write_proc(root, 900, "kworker/u16:9", utime=0, cmdline=b"")
  clock = FakeClock()
  sampler = ProcSampler(root=root, clock=clock, tick_hz=100)
  sampler.sample()
  clock.advance(5.0)
  for pid in range(500, 700):
    _bump(root, pid, 10 + pid % 7)      # keep every process above the 1% floor
  _bump(root, 900, 450)
  sampler.sample()
  payload = sampler.payload(budget=6000)
  assert len(json.dumps(payload).encode()) <= 6000
  assert any(row["k"] for row in payload["procs"])   # kernel outlier survived the trim
  assert payload["dropped"] > 0


def test_pump_edge_flush_and_throttle(fake_proc):
  clock = FakeClock()
  sampler = ProcSampler(root=fake_proc, clock=clock, tick_hz=100)
  flushes = []
  watch = LagWatch(sampler=sampler, on_flush=flushes.append, flush_min_interval=60.0)

  for _ in range(3):                    # no lag: sampling, zero flushes
    clock.advance(5.0)
    watch.pump()
  assert flushes == []

  watch.note_lag()                      # rising edge -> immediate flush
  clock.advance(5.0)
  watch.pump()
  assert len(flushes) == 1

  clock.advance(30.0)                   # lagging persists < 60 s -> throttled
  watch.pump()
  assert len(flushes) == 1

  clock.advance(35.0)                   # past the interval while still lagging -> one more
  watch.pump()
  assert len(flushes) == 2

  watch.note_ok()                       # cleared: no flush even though interval elapsed
  clock.advance(120.0)
  watch.pump()
  assert len(flushes) == 2

  watch.note_lag()                      # new edge re-arms immediate flush
  clock.advance(5.0)
  watch.pump()
  assert len(flushes) == 3


def test_pump_lag_source_polled(fake_proc):
  clock = FakeClock()
  sampler = ProcSampler(root=fake_proc, clock=clock, tick_hz=100)
  flushes = []
  state = {"lag": False}
  watch = LagWatch(sampler=sampler, on_flush=flushes.append, lag_source=lambda: state["lag"])
  clock.advance(5.0)
  watch.pump()
  assert flushes == []
  state["lag"] = True                     # simulated rk.lagging rising edge
  clock.advance(5.0)
  watch.pump()
  assert len(flushes) == 1
  clock.advance(30.0)
  watch.pump()
  assert len(flushes) == 1                # throttled while persisting


def test_kernel_outlier_survives_userspace_flood(tmp_path):
  # reviewer S2: >= TOP_N busy userspace must not truncate a saturating kernel thread
  root = tmp_path / "proc"
  root.mkdir()
  _write_proc_stat(root, {"cpu0": (1000, 900)})
  for pid in range(1000, 1015):           # 15 userspace, all above the 1% floor
    _write_proc(root, pid, "u", utime=0, cmdline=b"python\x00-m\x00mod\x00\x00")
  _write_proc(root, 1200, "ksoftirqd/5", utime=0, cmdline=b"")
  clock = FakeClock()
  sampler = ProcSampler(root=root, clock=clock, tick_hz=100)
  sampler.sample()
  clock.advance(5.0)
  for pid in range(1000, 1015):
    _bump(root, pid, 30)                  # 30 ticks / 500 = 6% each
  _bump(root, 1200, 480)                  # ksoftirqd pins core 5: 96%
  sampler.sample()
  payload = sampler.payload()
  userspace_rows = [r for r in payload["procs"] if not r["k"]]
  assert len(userspace_rows) == 12        # TOP_N cap
  krow = next(r for r in payload["procs"] if r["k"])
  assert krow["c"] == 96.0 and krow["n"] == "[ksoftirqd/5]"


def test_heartbeat_baseline(fake_proc):
  clock = FakeClock()
  sampler = ProcSampler(root=fake_proc, clock=clock, tick_hz=100)
  baselines = []
  watch = LagWatch(sampler=sampler, on_baseline=baselines.append, baseline_interval=300.0)
  clock.advance(5.0)
  watch.pump()
  assert baselines == []                  # offroad: silent
  watch.heartbeat(True)
  watch.pump()                            # onroad transition: baseline right away
  assert len(baselines) == 1
  clock.advance(120.0)
  watch.pump()
  assert len(baselines) == 1              # before interval: quiet
  clock.advance(180.0)                    # 300 s since last baseline
  watch.pump()
  assert len(baselines) == 2
  watch.heartbeat(False)                  # offroad again: quiet even past interval
  clock.advance(600.0)
  watch.pump()
  assert len(baselines) == 2


def test_install_guards(monkeypatch, tmp_path):
  from openpilot.starpilot import lag_watch
  monkeypatch.setenv("REPLAY", "1")
  assert lag_watch.install(object()) is None
  monkeypatch.delenv("REPLAY")
  monkeypatch.setenv("SIMULATION", "1")
  assert lag_watch.install(object()) is None
  monkeypatch.delenv("SIMULATION")
  import openpilot.common.hardware as hw
  monkeypatch.setattr(hw, "PC", True)
  assert lag_watch.install(object()) is None   # host runs never start sampler threads


def test_pump_survives_broken_sampler(tmp_path):
  class Boom(ProcSampler):
    def sample(self):
      raise RuntimeError("proc exploded")

  watch = LagWatch(sampler=Boom(root=tmp_path, clock=FakeClock()))
  watch.pump()   # must not propagate
