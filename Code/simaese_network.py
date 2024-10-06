import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=10)  # Output: (64, 119, 119)
        self.pool = nn.MaxPool2d(2, 2)  # Output: (64, 59, 59)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)  # Output: (128, 53, 53)
        self.pool = nn.MaxPool2d(2, 2)  # Output: (128, 26, 26)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)  # Output: (128, 23, 23)
        self.pool = nn.MaxPool2d(2, 2)  # Output: (128, 11, 11)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)  # Output: (256, 8, 8)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 4096)  # Update based on calculated size
        self.fc2 = nn.Linear(4096, 1)

    def forward_once(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        
        # Flatten
        x = x.view(x.size()[0], -1)  
        print(f"Shape after flattening: {x.shape}")  # Debug print
        
        x = F.relu(self.fc1(x))
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return torch.abs(output1 - output2)
