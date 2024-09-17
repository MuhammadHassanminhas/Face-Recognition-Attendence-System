import os
import cv2
import numpy as np
from torchvision import transforms 
from PIL import Image
import  torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize image to 64x64
    transforms.ToTensor()  # Convert image to a PyTorch tensor
])
class Image_Preprocessing():
    def __init__(self):
        self.data = []
        self.label = []
    def load_images(self,directory):
        for person in os.listdir(directory):
            person_dir = os.path.join(directory , person)
            if not os.path.isdir(person_dir):
                continue
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir , image_name)
                image = Image.open(image_path).convert('RGB')
                image = transform(image)
                image = image.to(device)
                self.data.append(image)
                self.label.append(person)
            self.data = torch.stack(self.data)
            return self.data , self.label

    
obj = Image_Preprocessing()
path = '/home/dread/Face Recognition/Data'
data , label = obj.load_images(path)

print(data.shape)
