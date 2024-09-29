import random
import torch
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    def __init__(self, positive_tensors, negative_tensors):
        self.positive_tensors = positive_tensors
        self.negative_tensors = negative_tensors

    def __len__(self):
        # Define the number of pairs based on the positive tensors
        return len(self.positive_tensors) * 10  # Example: adjust to the number of pairs you want

    def __getitem__(self, idx):
        # Randomly select a positive pair
        img_a = self.positive_tensors[idx % len(self.positive_tensors)]
        
        # Randomly choose a negative image
        img_b = random.choice(self.negative_tensors)

        # Label: 1 for same student (positive pair), 0 for different (negative pair)
        label = 1 if random.random() > 0.5 else 0

        return (img_a, img_b), label

