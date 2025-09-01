# main.py
import torch
from torch.utils.data import DataLoader
from src.utils import setup_logging, collate_fn, evaluate_samples
from src.config import (
    BATCH_SIZE, NUM_WORKERS, EPOCHS, LEARNING_RATE, DEVICE,
    GROUP_SIZE, CLIP_COEF, ENTROPY_COEF, UPDATE_EPOCHS,
    ROLLOUT_BATCH_SIZE, MINIBATCH_SIZE, MAX_NEW_TOKENS, ITERATIONS_PER_EPOCH,
    GRPO_ITERATIONS, MODEL_NAME, MODEL_REVISION, TRUST_REMOTE_CODE, 
    TRAIN_SUBSET_SIZE, GRPO_EPOCHS, USE_ACCELERATOR, GRPO_LEARNING_RATE,
    TENSORBOARD_LOG, POLICY_UPDATE
    
)
import logging
logger = logging.getLogger(__name__)

# from src.data_loader import load_and_merge_datasets, CirclesQADataset
from src.data_loader_nohf import load_and_merge_datasets, CirclesQADataset



from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

def load_pretrained_model(CKPT_PATH, DEVICE):
    
    """
    Initializes and returns the model and processor.
    """
    logger.info("Initializing model and processor...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     CKPT_PATH,
    #     trust_remote_code=True,
    #     revision=MODEL_REVISION
    # ).to(device)
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        trust_remote_code=True,
        revision=MODEL_REVISION
    )
    base_cfg = AutoConfig.from_pretrained(
        MODEL_NAME, #"microsoft/Florence-2-base-ft",
        trust_remote_code=True,
        revision=MODEL_REVISION, #"refs/pr/6",
    )   
    base_cfg.vocab_size = len(processor.tokenizer)
    base_cfg.text_config.vocab_size = len(processor.tokenizer)
    
    
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_PATH,
        config=base_cfg,                 # ← key line: supplies correct vision_config
        trust_remote_code=True,
        local_files_only=True,
        ).to(DEVICE)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # if not MODEL_VISION_TOWER_TRAIN:
    #     logger.info("Vision tower is frozen.")
    #     for param in model.vision_tower.parameters():
    #         #param.is_trainable = False
    #         param.requires_grad = False
        
    logger.info("Model and processor initialized.")
    return model, processor

if __name__ == "__main__":
    dataset_dict = load_and_merge_datasets(
        data_dir="/dms/workspace_2025/vuthede/VLM/RL-VLM/notebooks/circles_dataset",
        prefix="<QA><CirclesQA>",
        answers_options="full"
    )
    logger.info("Dataset loaded successfully.")

    model, processor = load_pretrained_model("/dms/workspace_2025/vuthede/VLM/RL-VLM/checkpoint",DEVICE)  
    model_to_test = model.module if hasattr(model, "module") else model
    bos_token_id = processor.tokenizer.bos_token_id  # e.g., 0
    eos_token_id = processor.tokenizer.eos_token_id  # e.g., 2)
    print(f'bos and eos tokens id :{bos_token_id} . {eos_token_id}')
    # train_acc = evaluate_samples(
    #                 model_to_test, processor, DEVICE, dataset_dict, "train",
    #                 prefix="<QA><CirclesQA>",
    #                 N=None, # default is len of dataset 
    #                 Q="Is the number of circles odd or even?",
    #                 print_outputs=True,
    #                 logger=logger,
    #             )
    # print("Train acc", train_acc)
    
    val_acc = evaluate_samples(
                    model_to_test, processor, DEVICE, dataset_dict, "validation",
                    prefix="<QA><CirclesQA>",
                    N=None, # default is len of dataset 
                    Q="Is the number of circles odd or even?",
                    print_outputs=False,
                    logger=logger,
                )
    print("Val acc", val_acc)
    test_acc = evaluate_samples(
                    model_to_test, processor, DEVICE, dataset_dict, "test",
                    prefix="<QA><CirclesQA>",
                    N=None, # default is len of dataset 
                    Q="Is the number of circles odd or even?",
                    print_outputs=False,
                    logger=logger,
                )
    print("Test acc", test_acc)