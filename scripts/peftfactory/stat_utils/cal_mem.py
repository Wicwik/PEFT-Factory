import wandb
import pandas as pd
from tqdm import tqdm

import numpy as np

api = wandb.Api()

methods = ["lora", "prefix-tuning", "prompt-tuning", "p-tuning", "lntuning", "ia3"]

mean_memory_usage = {}

for m in methods:
    print(f"Processing method: {m}")
    project = f""
    runs = api.runs(project)

    max_values = []

    for run in tqdm(runs):
        history = run.history(stream="events")  # load just 1 row
    
        if history.empty or "system.gpu.0.memoryAllocatedBytes" not in history.columns:
            continue
    
        max_val = history["system.gpu.0.memoryAllocatedBytes"].max()

        max_values.append({"run": run.name.split("/")[-1], "id": run.id, "max_memory_usage": max_val})

    df = pd.DataFrame(max_values)
    mean_memory_usage[m] = np.round(df["max_memory_usage"].mean()/1e9, 2)


df_mem = pd.DataFrame.from_dict(mean_memory_usage, orient="index", columns=["mean_memory_usage_GB"])
print(df_mem)