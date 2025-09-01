import json
import os
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
import logging
from datasets import DatasetDict
import torchvision.transforms as T
import torch
import numpy as np
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CirclesQADataset(TorchDataset):
    def __init__(self, data_dir, split, prefix, answers_options="full"):
        """
        Initialize dataset from generated JSON and image files.
        
        Args:
            data_dir (str): Base directory containing circles_dataset/ (e.g., '/dms/workspace_2025/vuthede/VLM/RL-VLM').
            split (str): Dataset split ('train', 'val', 'test', 'test_hard').
            prefix (str): Prefix for questions (e.g., 'CAPTION' or other).
            answers_options (str): Response format ('full' or 'short').
        """
        super().__init__()
        self.data_dir = os.path.join(data_dir, split)
        self.prefix = prefix
        self.answers_options = answers_options
        self.split = split
        
        # Load JSON annotations
        json_path = os.path.join(self.data_dir, "labels.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at {json_path}")
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            # print(f'Warning!!!! Jut get 100 samples for test')
            # self.data = self.data[:1000]
            # self.data = self.data[:500]
            
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        answer_short = example['answer_short']
        full_description = example['full_description_answer']
        num_circles = example['num_circles']
        
        # Handle question prefix
        if self.prefix == "CAPTION":
            question = self.prefix
        else:
            question = self.prefix + example['question']
        
        # Load image
        image_path = os.path.join(self.data_dir, os.path.basename(example['image']))
        image = Image.open(image_path)
        class AddGaussianNoise(torch.nn.Module):
            def __init__(self, std=5):  # std in [0..255] space; 5≈gentle noise
                super().__init__()
                self.std = std
            def forward(self, img: Image.Image):
                arr = np.array(img).astype(np.int16)
                noise = np.random.normal(0, self.std, arr.shape).astype(np.int16)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(arr)

        augment = T.Compose([
            # T.RandomHorizontalFlip(p=0.5),                               # flip (kept as is)
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0), # enhanced color jitter
            AddGaussianNoise(std=5),                                     # random noise (kept as is)
            # T.RandomGrayscale(p=0.1),                                    # convert to grayscale (10% chance)
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),        # adjust sharpness (30% chance)
            T.RandomAutocontrast(p=0.2),                                 # auto-contrast (20% chance)
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),             # mild blur
        ])
        
        if self.split=="train":
            image = augment(image)  # apply augmentations
        
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return question, full_description, image

def load_and_merge_datasets(data_dir, prefix="", answers_options="full"):
    """
    Load and merge datasets from JSON and image files.
    
    Args:
        data_dir (str): Base directory containing circles_dataset/.
        prefix (str): Prefix for questions.
        answers_options (str): Response format ('full' or 'short').
        train_subset_size (int, optional): Subset size for train.
        test_subset_size (int, optional): Subset size for test.
        test_hard_subset_size (int, optional): Subset size for test_hard.
    
    Returns:
        DatasetDict: Dictionary containing datasets for each split.
    """
    logger.info(f"Loading datasets from {data_dir}/circles_dataset...")
    
    # Define splits and their corresponding dataset names
    splits = {
        "train": "train",
        "validation": "val",
        "test": "test",
        "test_hard_number": "test_hard"
    }
    
    dataset_dict = {}
    for dataset_name, split in splits.items():
        logger.info(f"Loading {split} dataset...")
        dataset = CirclesQADataset(
            data_dir=data_dir,
            split=split,
            prefix=prefix,
            answers_options=answers_options,
            
        )
        dataset_dict[dataset_name] = dataset
        logger.info(f"Loaded {split} dataset with {len(dataset)} examples.")
    
    logger.info("Datasets loaded and merged.")
    return DatasetDict(dataset_dict)

# Example usage
if __name__ == "__main__":
    data_dir = "/dms/workspace_2025/vuthede/VLM/RL-VLM/notebooks/circles_dataset"
    dataset_dict = load_and_merge_datasets(
        data_dir=data_dir,
        prefix="<QA><CirclesQA>",
        answers_options="full"
    )
    print(dataset_dict)
    # Access a sample
    question, answer, image = dataset_dict["train"][0]
    print(f"Sample from train: Question={question}, Answer={answer}, Image={image}")