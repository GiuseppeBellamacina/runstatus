import json
import types

from runstatus import RunStatus
from runstatus.hf import RunStatusCallback
from runstatus.wandb_hook import attach_wandb


def read(p):
    return json.loads(p.read_text())


def test_log_progress_and_best(tmp_path):
    p = tmp_path / "s.json"
    s = RunStatus(p, total_steps=100, throttle=0, kind="eval")
    s.set(phase="grpo")
    s.log(10, loss=0.5, reward=0.2)
    s.log(20, loss=0.7, reward=float("nan"))
    d = read(p)
    assert d["run"]["kind"] == "eval" and d["progress"]["pct"] == 20.0
    assert d["progress"]["step"] == 20 and d["metrics"]["latest"]["loss"] == 0.7 and d["run"]["phase"] == "grpo"
    assert d["metrics"]["latest"]["reward"] == 0.2 and "series" not in d
    assert d["metrics"]["best"]["loss"] == 0.5
    s.finish()
    d = read(p)
    assert d["run"]["state"] == "finished" and d["progress"]["pct"] == 100.0


def test_series_decimation(tmp_path):
    s = RunStatus(tmp_path / "s.json", throttle=0, max_points=50)
    for i in range(1000):
        s.log(i, loss=1.0)
    assert len(s.d["series"]["loss"]) <= 50


def test_failsafe_unwritable(tmp_path):
    blocker = tmp_path / "f"
    blocker.write_text("x")
    s = RunStatus(blocker / "sub" / "s.json")  # parent is a file: must not raise
    s.log(1, loss=1.0)
    s.finish()


def test_hf_callback(tmp_path):
    p = tmp_path / "s.json"
    s = RunStatus(p, throttle=0)
    cb = RunStatusCallback(s)
    st = types.SimpleNamespace(is_local_process_zero=True, max_steps=50, global_step=5, epoch=0.1)
    cb.on_train_begin(None, st, None)
    cb.on_log(None, st, None, logs={"loss": 1.5, "total_flos": 3, "epoch": 0.1})
    cb.on_train_end(None, st, None)
    d = read(p)
    assert d["progress"]["total_steps"] == 50 and d["metrics"]["latest"]["loss"] == 1.5 and d["run"]["state"] == "finished"


def test_wandb_hook(tmp_path):
    p = tmp_path / "s.json"
    s = RunStatus(p, throttle=0)
    calls = []
    fake = types.SimpleNamespace(run=types.SimpleNamespace(id="abc", url=None, settings=None),
                                 log=lambda data=None, *a, **kw: calls.append(data))
    attach_wandb(s, fake)
    fake.log({"loss": 2.0}, step=7)
    d = read(p)
    assert calls == [{"loss": 2.0}] and d["progress"]["step"] == 7 and d["wandb"]["id"] == "abc"


def test_rl_rewards_stats_and_best(tmp_path):
    s = RunStatus(tmp_path / "s.json", throttle=0, window=2)
    for i, r in enumerate([0.0, 1.0, 0.5], start=1):
        s.log(i, reward=r, kl=0.1)
    m = read(tmp_path / "s.json")["metrics"]
    assert m["mean"]["reward"] == 0.5 and m["moving"]["reward"] == 0.75
    assert m["best"]["reward"] == 1.0 and "kl" not in m["best"]
    assert m["stats"]["reward"] == {"min": 0.0, "max": 1.0}


def test_include_exclude_rename_best_rule(tmp_path):
    s = RunStatus(tmp_path / "s.json", throttle=0, exclude=["debug/*"], rename={"learning_rate": "lr"},
                  best={"custom": "max"}, series=["loss"])
    s.log(1, **{"debug/x": 1.0, "learning_rate": 0.1, "custom": 3.0, "loss": 2.0})
    s.log(2, custom=2.0, loss=1.0)
    s.finish()  # forces the series sidecar
    d = read(tmp_path / "s.json")
    assert "debug/x" not in d["metrics"]["latest"] and d["metrics"]["latest"]["lr"] == 0.1
    assert d["metrics"]["best"]["custom"] == 3.0 and list(read(tmp_path / "s.json.series.json")["series"]) == ["loss"]


def test_extra_events_eval_and_context_manager(tmp_path):
    p = tmp_path / "s.json"
    try:
        with RunStatus(p, throttle=0, extra={"lr": 1e-4}) as s:
            s.set(extra={"seed": 1}, eval_active=True, eval_progress="3/10", phase="eval")
            s.event("hello")
            raise ValueError("boom")
    except ValueError:
        pass
    d = read(p)
    assert d["extra"] == {"lr": 1e-4, "seed": 1} and d["run"]["state"] == "failed"
    assert d["eval"]["active"] is False and "boom" in d["note"]
    assert any(e["msg"] == "hello" for e in d["events"])


def test_hf_transform_eval_and_save(tmp_path):
    p = tmp_path / "s.json"
    s = RunStatus(p, throttle=0)
    cb = RunStatusCallback(s, phase="grpo", transform=lambda logs, st: {**logs, "double": logs["loss"] * 2})
    st = types.SimpleNamespace(is_local_process_zero=True, max_steps=10, global_step=3, epoch=0.3)
    cb.on_train_begin(None, st, None)
    cb.on_log(None, st, None, logs={"loss": 1.0})
    cb.on_prediction_step(None, st, None)
    assert read(p)["eval"]["active"] is True
    cb.on_evaluate(None, st, None, metrics={"eval_loss": 0.4})
    cb.on_save(None, st, None)
    d = read(p)
    assert d["metrics"]["latest"]["double"] == 2.0 and d["eval"]["active"] is False
    assert d["metrics"]["best"]["eval_loss"] == 0.4 and d["run"]["phase"] == "grpo"
    assert any("checkpoint" in e["msg"] for e in d["events"])


def test_sidecar_and_reset_series(tmp_path):
    p = tmp_path / "s.json"
    s = RunStatus(p, throttle=0, total_steps=100)
    s.set(phase="sft")
    for i in (1, 2, 3):
        s.log(i, loss=1.0 / i)
    s.finish()
    side = read(tmp_path / "s.json.series.json")
    assert [pt[0] for pt in side["series"]["loss"]] == [1, 2, 3] and side["total_steps"] == 100
    assert read(p)["series_file"] == "s.json.series.json"
    s2 = RunStatus(p, throttle=0)
    s2.log(5, loss=1.0)
    s2.set(phase="grpo", reset_series=True)
    s2.log(1, loss=0.5)
    s2.finish()
    assert read(tmp_path / "s.json.series.json")["series"]["loss"] == [[1, 0.5]]


def test_default_path_follows_slurm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNSTATUS_PATH", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    RunStatus(throttle=0)
    assert (tmp_path / "logs" / "run_status-4242.json").is_file()
    monkeypatch.setenv("RUNSTATUS_PATH", str(tmp_path / "x.json"))
    RunStatus(throttle=0)
    assert (tmp_path / "x.json").is_file()
