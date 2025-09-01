# CUDA_VISIBLE_DEVICES=7 taskset --cpu-list 60-70 python3 main.py
source ~/proxy.sh # for model hugging face, otherwise it hangover
CUDA_VISIBLE_DEVICES=7 taskset --cpu-list 50-70 python3 main_trainer.py
