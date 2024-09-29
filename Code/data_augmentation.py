import os
from PIL import Image
import torch
from torchvision import transforms

# Define the path to your 'positive' folder
positive_folder = '/home/dread/Face Recognition/Data/Positive'

# Augmentation transformations (without normalization)
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),  # Reduced rotation angle
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Less aggressive ColorJitter
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),  # Less extreme resizing
    transforms.ToTensor()
])

# Loop through each student's folder and apply augmentations
for student in os.listdir(positive_folder):
    student_folder = os.path.join(positive_folder, student)
    
    for img_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_name)
        image = Image.open(img_path)
        
        # Apply augmentations (let's create 5 augmented images per student)
        for i in range(5):
            augmented_image = augment_transform(image)  # Apply augmentation
            
            # Save the augmented image
            save_path = os.path.join(student_folder, f"augmented_{i}_{img_name}")
            transforms.ToPILImage()(augmented_image).save(save_path)

print("Augmentation completed without extreme color distortions.")

