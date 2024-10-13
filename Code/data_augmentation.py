import torchvision.transforms as transforms
from PIL import Image

class DataAugmentation:
    def __init__(self):
        # Define the augmentation transformations (e.g., resizing, flipping, etc.)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.ToTensor()
        ])

    def augment(self, image):
        # Apply transformations on the image
        augmented_image = self.transform(image)
        return augmented_image

