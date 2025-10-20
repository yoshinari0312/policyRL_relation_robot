import subprocess
import wandb

wandb.init(project="rl-convo", name="gpu-monitor")

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
    capture_output=True,
    text=True,
    check=True,
)
for line in result.stdout.strip().splitlines():
    gpu_id, util = line.split(", ")
    wandb.log({f"gpu/{gpu_id}/util": float(util)}, commit=False)
wandb.log({}, commit=True)