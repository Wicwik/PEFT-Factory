import numpy as np

llama_base_flops = 8.38

methods = ["lora", "prefix-tuning", "prompt-tuning", "p-tuning", "lntuning", "ia3"]
methods_parameters_map = {
    "ia3": 196608,
    "prompt-tuning": 409600,
    "prefix-tuning": 34177536,
    "p-tuning": 53130752,
    "lora": 14680064,
    "lntuning": 266240,
}

methods_flops_map = {
    "ia3": 8.69 - llama_base_flops,
    "prompt-tuning": 10.05 - llama_base_flops,
    "prefix-tuning": 8.39 - llama_base_flops,
    "p-tuning": 10.06 - llama_base_flops,
    "lora": 8.38 - llama_base_flops,
    "lntuning": 8.38 - llama_base_flops,
}

methods_memory_map = {
    "ia3": 28.41,
    "prompt-tuning": 42.47,
    "prefix-tuning": 27.61,
    "p-tuning": 34.53,
    "lora": 27.75,
    "lntuning": 27.43,
}

performance_map = {
    "ia3": 74.7,
    "prompt-tuning": 50,
    "prefix-tuning": 45.9,
    "p-tuning": 51.7,
    "lora": 80.1,
    "lntuning": 77.8,
}

def cost(m, c):
    return (1 + m/c)**-1

def psqp(p, cost_params, cost_flops, cost_memory):
    return p * cost_params * cost_flops * cost_memory


# c_p = np.median(list(methods_parameters_map.values()))
# c_p = np.exp(np.mean(np.log(list(methods_parameters_map.values()))))
# c_p = np.max(list(methods_parameters_map.values()))
c_p = 5e8 # 
c_f = 10 # such slowdown would mean that its better to use the base model
c_m = 94 # h100 nvl has 94GB

psqp_values = {}

for method in methods:
    cost_params = cost(methods_parameters_map[method], c_p)
    print(f"{method}: {methods_parameters_map[method]} params, cost for params: {cost_params:.2f}")

    cost_flops = cost(methods_flops_map[method], c_f)
    print(f"{method}: {methods_flops_map[method]:.2f} TFLOPs, cost for flops: {cost_flops:.2f}")

    cost_memory = cost(methods_memory_map[method], c_m)
    print(f"{method}: {methods_memory_map[method]:.2f} GB, cost for memory: {cost_memory:.2f}")

    psqp_values[method] = np.round(psqp(performance_map[method], cost_params, cost_flops, cost_memory), 1)


print(psqp_values)