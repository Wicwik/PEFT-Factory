# Copyright 2025 the PEFTFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json

import pandas as pd


# models = ["gemma-3-1b-it", "llama-3-8b-instruct", "mistral-7b-instruct"]
models = ["llama-3-8b-instruct"]
# methods = ["base", "ia3", "lora", "lntuning", "prompt-tuning", "p-tuning"]
methods = ["prefix-tuning"]
# methods = ["base"]
datasets = [
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "rte",
    "cola",
    "record",
    "multirc",
    "boolq",
    "wic",
    "wsc",
    "cb",
    "copa",
]
# datasets = ["mmlu", "piqa", "siqa", "hellaswag", "winogrande", "openbookqa", "math_qa", "gsm8k", "svamp", "conala", "codealpacapy", "apps"]
# datasets = ["record", "multirc", "boolq", "wic", "wsc", "cb", "copa"]


def get_single_result(results, dataset):
    print(dataset)
    if "macro_f1" in results:
        return results["macro_f1"]
    elif "pearsonr" in results:
        return results["pearsonr"]
    elif dataset in ["gsm8k", "svamp"]:
        return results["accuracy"]
    elif dataset in ["conala", "codealpacapy", "apps"]:
        return results["codebleu"]
    else:
        return results["f1"]


def get_results_from_jsonl(eval_dir):
    results = {}
    with open(f"{eval_dir}/results.jsonl") as json_file:
        for line in json_file:
            results.update(json.loads(line))

    return results


for m in models:
    print(f"Model {m}")

    results = {}
    for pm in methods:
        print(f"Method {pm}")
        results[pm] = {}
        for d in datasets:
            print(f"Dataset {d}")
            glob_res = glob.glob(f"saves/{pm}/{m}/eval_{d}*")

            if not glob_res:
                continue

            try:
                results[pm][d] = get_single_result(get_results_from_jsonl(sorted(glob_res)[-1]), d) * 100
            except FileNotFoundError:
                continue

    results_df = pd.DataFrame(results).T
    print(
        results_df.to_latex(
            float_format="%.1f", caption="Performance across tasks and tuning methods", label="tab:results"
        )
    )
