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

    def __len__(self):
      return len(self.positive_images)

    def __getitem__(self, idx):
        # Get a positive image
        pos_image, pos_filename = self.positive_images[idx % len(self.positive_images)]
        # Get a negative image, loop through negatives
        neg_image, neg_filename = self.negative_images[idx % len(self.negative_images)]

        # Resize and transform both images
        if self.transform:
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)

        # Check if the transformed images match the target size
        self.check_image_size(pos_image, pos_filename)
        self.check_image_size(neg_image, neg_filename)

        return pos_image, neg_image  # Return both images as a tuple

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
