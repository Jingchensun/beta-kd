bash 2_finetune.sh 1 align-kd equal jsun39/AlignKD-Pretrain-1246k
bash scripts/benchmark.sh outputs-finetune/align-kd-equal/ eval-results/align-kd-equal

bash 2_finetune.sh 1 align-kd task outputs-pretrain/align-kd-task/
bash scripts/benchmark.sh outputs-finetune/align-kd-task/ eval-results/align-kd-task


bash 2_finetune.sh 1 align-kd instance outputs-pretrain/align-kd-instance/
bash scripts/benchmark.sh outputs-finetune/align-kd-instance/ eval-results/align-kd-instance
