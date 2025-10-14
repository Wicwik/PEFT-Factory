datasets=(wsc)
# datasets=(cb copa wsc svamp conala rte mrpc openbookqa wic stsb cola gsm8k siqa math_qa winogrande sst2)
# peft_methods=(prefix-tuning prompt-tuning p-tuning lora lntuning ia3)
peft_methods=(prefix-tuning)
models=(llama-3-8b-instruct)
seeds=(42 123 456 789 101112)
EPOCHS=30

for s in ${seeds[@]};
do
    for d in ${datasets[@]};
    do
        for m in ${models[@]};
        do
            for pm in ${peft_methods[@]};
            do
                TIMESTAMP=`date +%s`
                OUTPUT_DIR="saves/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}"
                DATASET="${d}"
                SEED="${s}"
                WANDB_PROJECT="peft-factory-${pm}"
                WANDB_NAME="${pm}_${m}_train_${d}_${s}_${TIMESTAMP}"


                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME EPOCHS
                envsubst < examples/peft/${pm}/${m}/train.yaml > ${OUTPUT_DIR}/train.yaml

                OUTPUT_DIR="saves/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}"
                WANDB_NAME="${pm}_${m}_eval_${d}_${s}_${TIMESTAMP}"
                ADAPTER="saves/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}"

                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR WANDB_NAME ADAPTER
                envsubst < examples/peft/${pm}/${m}/eval.yaml > ${OUTPUT_DIR}/eval.yaml

                llamafactory-cli train saves/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}/train.yaml
                llamafactory-cli train saves/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}/eval.yaml
                python scripts/peftfactory/compute_metrics.py saves/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP} ${d}
            done
        done
    done
done