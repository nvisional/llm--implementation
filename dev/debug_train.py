import debugpy
debugpy.listen(("localhost", 9501))
print("Waiting for debugger attach on port 9501 ...")
debugpy.wait_for_client()
print("Debugger attached!")

# ---- inject tiny training args before base_train.py parses sys.argv ----
import sys
sys.argv = [
    "base_train",
    "--depth=4",
    "--max-seq-len=64",       # tiny sequence: easy to inspect tensors
    "--device-batch-size=1",
    "--total-batch-size=64",
    "--num-iterations=3",     # only 3 steps, enough to see one full loop
    "--eval-every=-1",
    "--core-metric-every=-1",
    "--sample-every=-1",
    "--window-pattern=L",     # full context (no sliding window warning on MPS)
]

# ---- now just run base_train as if it were the main script ----
import runpy
runpy.run_module("scripts.base_train", run_name="__main__")
