import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, positive_path, negative_path, transform=None):
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.transform = transform
        self.positive_images = self.load_images(positive_path)
        self.negative_images = self.load_images(negative_path)

    def load_images(self, path):
        images = []
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):  # Ensure it's a directory
                for img_file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
        return images

    def __len__(self):
        return min(len(self.positive_images), len(self.negative_images))

    def __getitem__(self, idx):
        pos_image = self.positive_images[idx]
        neg_image = self.negative_images[idx]  # Loop through negatives

        if self.transform:
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)

        return pos_image, neg_image

def create_dataloaders(positive_path, negative_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Resize((128, 128)),  # Resize to a fixed size
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    dataset = ImageDataset(positive_path, negative_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, dataloader
