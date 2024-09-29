import os
from PIL import Image
import torch
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, positive_folder, negative_folder, device=None, batch_size=5):
        self.positive_folder = positive_folder
        self.negative_folder = negative_folder
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.positive_tensors = []
        self.negative_tensors = []

    def preprocess_and_move_to_gpu(self, image_path):
        image = Image.open(image_path)
        tensor_image = self.preprocess_transform(image)
        tensor_image = tensor_image.to(self.device)
        return tensor_image

    def process_images_in_folder(self, folder_path):
        image_tensors = []
        for person in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person)
            
            if os.path.isdir(person_folder):
                for img_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_name)
                    
                    if os.path.isfile(img_path):
                        processed_image = self.preprocess_and_move_to_gpu(img_path)
                        image_tensors.append(processed_image)
                        print(f"Processed image for {img_name} in {folder_path}. Shape: {processed_image.shape}")

                        # Clear the cache if the batch size is reached
                        if len(image_tensors) >= self.batch_size:
                            torch.cuda.empty_cache()
                            image_tensors.clear()  # Clear the current batch list
        
        return image_tensors

    def process_all_images(self):
        print("Processing Positive images...")
        self.positive_tensors = self.process_images_in_folder(self.positive_folder)
        
        print("Processing Negative images...")
        self.negative_tensors = self.process_images_in_folder(self.negative_folder)
        
        print("Preprocessing and GPU transformation for both Positive and Negative images completed.")
    
    def get_tensors(self):
        return self.positive_tensors, self.negative_tensors

