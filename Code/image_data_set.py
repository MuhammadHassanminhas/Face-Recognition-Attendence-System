import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Define your target image size
TARGET_SIZE = (128, 128)  # Adjust this as needed

class ImageDataset(Dataset):
    def __init__(self, positive_path, negative_path, transform=None):
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.transform = transform
        self.positive_images = self.load_images(positive_path)
        self.negative_images = self.load_images(negative_path)

        # Lists to store tensors for positive and negative images
        self.positive_tensors = []
        self.negative_tensors = []

        # Preprocess and convert images to tensors
        self.preprocess_images()

    def load_images(self, path):
        images = []
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append((img, img_file))  # Store both the image and its filename
        return images

    def preprocess_images(self):
        for img, filename in self.positive_images:
            img_tensor = self.transform(img) if self.transform else img
            self.check_image_size(img_tensor, filename)
            self.positive_tensors.append(img_tensor)

        for img, filename in self.negative_images:
            img_tensor = self.transform(img) if self.transform else img
            self.check_image_size(img_tensor, filename)
            self.negative_tensors.append(img_tensor)

    def __len__(self):
        return max(len(self.positive_tensors), len(self.negative_tensors))

    def __getitem__(self, idx):
        # Get a positive image tensor
        pos_image = self.positive_tensors[idx % len(self.positive_tensors)]
        # Get a negative image tensor, loop through negatives
        neg_image = self.negative_tensors[idx % len(self.negative_tensors)]

        return pos_image, neg_image

    def check_image_size(self, image, filename):
        # Get the size of the transformed image
        size = image.shape[1:]  # Exclude batch dimension
        if size != TARGET_SIZE:
            print(f"Warning: Image '{filename}' does not match target size: {size}. Expected: {TARGET_SIZE}")

def create_dataloaders(positive_path, negative_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Resize(TARGET_SIZE),  # Resize to a fixed size
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    dataset = ImageDataset(positive_path, negative_path, transform=transform)
    
    # Create a DataLoader that provides both positive and negative images
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# Example usage
if __name__ == "__main__":
    positive_path = '/home/dread/Face Recognition/Data/Positive'
    negative_path = '/home/dread/Face Recognition/Data/Negative'
    
    dataloader = create_dataloaders(positive_path, negative_path, batch_size=32)

    for pos_images, neg_images in dataloader:
        # Do something with pos_images and neg_images
        print(f"Positive batch shape: {pos_images.shape}, Negative batch shape: {neg_images.shape}")  # Example to show the shape of the batches
