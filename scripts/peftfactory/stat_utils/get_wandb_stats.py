import wandb
import sys

def get_max_gpu_memory(run_id, entity=None, project=None):
    """
    Fetch the maximum GPU memory allocation (in bytes) from a wandb run.
    Looks for the 'system.gpu.memoryAllocated' metric.
    """
    api = wandb.Api()
    if entity and project:
        run = api.run(f"{entity}/{project}/{run_id}")
    else:
        run = api.run(run_id)
    history = run.history(samples=10000)
    if "system.gpu.memoryAllocated" in history.columns:
        max_mem = history["system.gpu.memoryAllocated"].max()
        print(f"Max GPU memory allocated for run {run_id}: {max_mem} bytes")
        return max_mem
    else:
        print("No GPU memory allocation data found for this run.")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_wandb_stats.py <run_id> [entity] [project]")
        sys.exit(1)
    run_id = sys.argv[1]
    entity = sys.argv[2] if len(sys.argv) > 2 else None
    project = sys.argv[3] if len(sys.argv) > 3 else None