"""Simulated RL run: python examples/fake_rl_train.py [out.json]  then  python -m runstatus show out.json"""
import math
import random
import sys

from runstatus import RunStatus

out = sys.argv[1] if len(sys.argv) > 1 else "logs/run_status.json"
random.seed(0)
st = RunStatus(out, kind="train", name="grpo-demo", throttle=0, method="grpo",
               model={"family": "llm", "name": "demo", "params": "0.5B"}, stages=["sft", "grpo", "eval"],
               badges=["LoRA", {"label": "4-bit", "icon": "gpu"}], extra={"lr": 1e-5})
st.stage("sft", total_steps=100)
for step in range(10, 101, 10):
    st.log(step, loss=2 / math.sqrt(step))
st.stage("grpo", total_steps=500)
for step in range(10, 351, 10):
    st.log(step, loss=1 / math.sqrt(step) + random.random() * 0.02,
           reward=min(1, step / 400 + random.random() * 0.1), **{"rewards/format": 0.9, "kl": 0.01})
    if step % 100 == 0:
        st.set(eval_active=True)
        st.log(step, eval_loss=1 / math.sqrt(step))
        st.set(eval_active=False)
        st.event(f"checkpoint saved at step {step}")
# no finish(): the run is still "running", like a real job in progress
