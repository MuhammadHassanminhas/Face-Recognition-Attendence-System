import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import encoding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Image_Classification(torch.nn.Module):
    def __init__(self , num_classes):
        super(Image_Classification , self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)  # num_classes is the number of people
        
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutions, ReLU, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



num_classes = len(encoding.label_encode.classes_)
obj = Image_Classification(num_classes).to(device)



