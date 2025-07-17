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

import argparse
import json

import evaluate
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score


def string_to_float(string, default=-1.0):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def check_data_state(preds, targets):
    assert len(preds) == len(targets)


def binary_reverse(targets, labels):
    return [labels[0] if target == labels[1] else labels[1] for target in targets]


def em(preds, targets, labels):
    check_data_state(preds, targets)

    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    return {"exact_match": np.sum(preds == targets) / preds.size}


# def f1(preds, targets, labels):
#     check_data_state(preds, targets)

#     preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

#     invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

#     preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

#     return {"f1": f1_score(targets, preds, labels=labels, pos_label=labels[0])}


# def macro_f1(preds, targets, labels):
#     check_data_state(preds, targets)

#     preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

#     invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

#     preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

#     return {"macro_f1": f1_score(targets, preds, labels=labels, average="macro")}


def f1(preds, targets, labels):
    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

    label_to_id = {label: idx for idx, label in enumerate(labels)}

    # print(label_to_id)

    targets = list(map(label_to_id.get, targets))
    preds = list(map(label_to_id.get, preds))

    # print(preds, targets)

    return {"f1": f1_score(targets, preds)}


def macro_f1(preds, targets, labels):
    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

    label_to_id = {label: idx for idx, label in enumerate(labels)}

    # print(label_to_id)

    targets = list(map(label_to_id.get, targets))
    preds = list(map(label_to_id.get, preds))

    # print(preds, targets)

    return {"macro_f1": f1_score(targets, preds, average="macro")}


def pearsonr(preds, targets, labels):
    metric = evaluate.load("pearsonr")

    targets = [string_to_float(t) for t in targets]
    preds = [string_to_float(p) for p in preds]

    return metric.compute(predictions=preds, references=targets)


def spearmanr(preds, targets, labels):
    metric = evaluate.load("spearmanr")

    targets = [string_to_float(t) for t in targets]
    preds = [string_to_float(p) for p in preds]

    return metric.compute(predictions=preds, references=targets)


def record(preds):
    dataset = load_dataset("rbelanec/record", split="validation")
    metric = evaluate.load("super_glue", "record")

    predictions = [{"idx": dataset[i]["idx"], "prediction_text": p} for i, p in enumerate(preds)]

    references = [{"idx": d["idx"], "answers": d["answers"]} for d in dataset]

    return metric.compute(predictions=predictions, references=references)


# def multirc(preds, targets):
#     dataset = load_dataset("rbelanec/multirc", split="validation")
#     metric = evaluate.load("super_glue", "multirc")

#     lm = {"true": 1, "false": 0}
#     reverse = {"True": 0, "False": 1}

#     predictions = [{"idx": dataset[i]["idx"], "prediction": lm.get(p.split(" ")[0].lower(), reverse[targets[i]])} for i, p in enumerate(preds)]
#     references = [lm[t.lower()] for t in targets]

#     print(predictions, references)

#     return metric.compute(predictions=predictions, references=references)


DATASET_TO_METRIC_MAPPING = {
    "mnli": {"metrics": [macro_f1, em], "labels": ["entailment", "neutral", "contradiction"]},
    "qqp": {"metrics": [f1, em], "labels": ["not_duplicate", "duplicate"]},
    "qnli": {"metrics": [f1, em], "labels": ["entailment", "not_entailment"]},
    "sst2": {"metrics": [f1, em], "labels": ["negative", "positive"]},
    "stsb": {"metrics": [pearsonr, spearmanr], "labels": []},
    "mrpc": {"metrics": [f1, em], "labels": ["not_equivalent", "equivalent"]},
    "rte": {"metrics": [f1, em], "labels": ["entailment", "not_entailment"]},
    "cola": {"metrics": [f1, em], "labels": ["unacceptable", "acceptable"]},
    "record": {"metrics": [record], "labels": []},
    "multirc": {"metrics": [f1, em], "labels": ["false", "true"]},
    "boolq": {"metrics": [f1, em], "labels": ["false", "true"]},
    "wic": {"metrics": [f1, em], "labels": ["false", "true"]},
    "wsc": {"metrics": [f1, em], "labels": ["false", "true"]},
    "cb": {"metrics": [macro_f1, em], "labels": ["entailment", "contradiction", "neutral"]},
    "copa": {"metrics": [f1, em], "labels": ["choice1", "choice2"]},
}

argparse_parser = argparse.ArgumentParser(
    prog="Compute metrics.",
    description="Compute metrics for single model.",
)

argparse_parser.add_argument("eval_dir", help="Directory created during evaluation.")
argparse_parser.add_argument("dataset", help="Dataset used during evaluation.")

args = argparse_parser.parse_args()

eval_dir = args.eval_dir
dataset = args.dataset

eval_samples = []
with open(f"{eval_dir}/generated_predictions.jsonl") as json_file:
    for line in json_file:
        eval_samples.append(json.loads(line))

labels, predictions = [], []
for es in eval_samples:
    labels.append(es["label"].split("</think>\n\n")[-1].strip().lower())
    predictions.append(es["predict"].split("</think>\n\n")[-1].strip().lower())

# print(list(zip(predictions,labels)))

with open(f"{eval_dir}/results.jsonl", "w") as outfile:
    for metric in DATASET_TO_METRIC_MAPPING[dataset]["metrics"]:
        if dataset in ["record"]:
            result = metric(predictions)
        else:
            result = metric(predictions, labels, DATASET_TO_METRIC_MAPPING[dataset]["labels"])

        print(result)
        json.dump(result, outfile)
        outfile.write("\n")
