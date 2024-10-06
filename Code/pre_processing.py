import os
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Preprocessing:
    def __init__(self, positive_path, negative_path, batch_size=32, img_size=(128, 128)):
        self.transform = transforms.Compose([
            transforms.Resize(img_size),  # Resize images to 128x128
            transforms.Grayscale(),  # Convert images to grayscale
            transforms.ToTensor(),  # Convert images to Tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
        ])
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.batch_size = batch_size

    def load_data(self):
        positive_dataset = datasets.ImageFolder(root=self.positive_path, transform=self.transform)
        negative_dataset = datasets.ImageFolder(root=self.negative_path, transform=self.transform)
        
        positive_loader = DataLoader(positive_dataset, batch_size=self.batch_size, shuffle=True)
        negative_loader = DataLoader(negative_dataset, batch_size=self.batch_size, shuffle=True)

        return positive_loader, negative_loader
