datasets=(cola cb svamp copa)
peft_methods=(prefix-tuning prompt-tuning p-tuning lora lntuning ia3)
models=(llama-3-8b-instruct)
seeds=(42 123 456 789 101112)
EPOCHS=20

for s in ${seeds[@]};
do
    for d in ${datasets[@]};
    do
        for m in ${models[@]};
        do
            for pm in ${peft_methods[@]};
            do
                TIMESTAMP=`date +%s`
                OUTPUT_DIR="saves_stability/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}"
                DATASET="${d}"
                SEED="${s}"
                WANDB_PROJECT="peft-factory-stability-${pm}"
                WANDB_NAME="${pm}_${m}_train_${d}_${s}_${TIMESTAMP}"


                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME EPOCHS
                envsubst < examples/peft/${pm}/${m}/train.yaml > ${OUTPUT_DIR}/train.yaml

                OUTPUT_DIR="saves_stability/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}"
                WANDB_NAME="${pm}_${m}_eval_${d}_${s}_${TIMESTAMP}"
                ADAPTER="saves_stability/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}"

                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR WANDB_NAME ADAPTER
                envsubst < examples/peft/${pm}/${m}/eval.yaml > ${OUTPUT_DIR}/eval.yaml

                sbatch --job-name ${pm}_${m}_stability_${d}_${s}_${TIMESTAMP} -o logs/${pm}_${m}_stability_${d}_${s}_${TIMESTAMP}.out -e logs/${pm}_${m}_stability_${d}_${s}_${TIMESTAMP}.err scripts/peftfactory/slurm/run_train_eval.sh saves_stability/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}/train.yaml saves_stability/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}/eval.yaml saves_stability/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP} ${d}

                sleep 1
            done
        done
    done
done