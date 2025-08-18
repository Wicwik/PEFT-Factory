#!/bin/bash

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

# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola record multirc boolq wic wsc cb copa)
# datasets=(record multirc boolq wic wsc cb copa)
# peft_methods=(ia3 prompt-tuning lora lntuning)
datasets=(mmlu piqa siqa hellaswag winogrande openbookqa math_qa gsm8k svamp conala codealpacapy apps)
peft_methods=(base)
models=(llama-3-8b-instruct)


for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            saves=(saves/${pm}/${m}/eval_${d}_*)

            EVAL_DIR="${saves[-1]}"

            python scripts/peftfactory/compute_metrics.py ${EVAL_DIR} ${d}
        done
    done
done
