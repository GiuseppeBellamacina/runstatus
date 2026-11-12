"""Simulated RL run: python examples/fake_rl_train.py [out.json]  then  python -m runstatus show out.json"""
import math
import random
import sys

from runstatus import RunStatus

out = sys.argv[1] if len(sys.argv) > 1 else "logs/run_status.json"
random.seed(0)
st = RunStatus(out, kind="rl", name="grpo-demo", total_steps=500, throttle=0,
               extra={"model": "demo-0.5b", "lr": 1e-5})
st.set(phase="grpo")
for step in range(10, 351, 10):
    st.log(step, loss=1 / math.sqrt(step) + random.random() * 0.02,
           reward=min(1, step / 400 + random.random() * 0.1), **{"rewards/format": 0.9, "kl": 0.01})
    if step % 100 == 0:
        st.set(eval_active=True)
        st.log(step, eval_loss=1 / math.sqrt(step))
        st.set(eval_active=False)
        st.event(f"checkpoint saved at step {step}")
# no finish(): the run is still "running", like a real job in progress
