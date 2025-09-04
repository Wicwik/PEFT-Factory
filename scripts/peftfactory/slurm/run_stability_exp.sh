datasets=(cb)
peft_methods=(prefix-tuning)
models=(llama-3-8b-instruct)
seeds=(42 123 456 789 101112)

for seeds in ${seeds[@]};
do
    for d in ${datasets[@]};
    do
        for m in ${models[@]};
        do
            for pm in ${peft_methods[@]};
            do
                TIMESTAMP=`date +%s`
                OUTPUT_DIR="saves_stability/${pm}/${m}/train_${d}_${TIMESTAMP}"
                DATASET="${d}"
                SEED="${seeds}"
                WANDB_PROJECT="peft-factory-stability-${pm}"
                WANDB_NAME="${pm}_${m}_train_${d}"

                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME

                envsubst < examples/peft/${pm}/${m}/train.yaml > ${OUTPUT_DIR}/train.yaml
                envsubst < examples/peft/${pm}/${m}/eval.yaml > ${OUTPUT_DIR}/eval.yaml

                sbatch --job-name ${pm}_${m}_train_${d}_${TIMESTAMP} -o logs/${pm}_${m}_train_${d}_${TIMESTAMP}.out -e logs/${pm}_${m}_train_${d}_${TIMESTAMP}.err scripts/peftfactory/slurm/run_train_eval.sh ${OUTPUT_DIR}/train.yaml ${OUTPUT_DIR}/eval.yaml

                sleep 1
            done
        done
    done
done