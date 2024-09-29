import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10)  # Input channels = 3 for RGB images
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward_one(self, x):
        # Pass through convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Pass through the fully connected layer
        x = F.relu(self.fc1(x))
        
        return x

    def forward(self, input1, input2):
        # Get the output for each image pair
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        # Compute the absolute difference between the two outputs
        diff = torch.abs(output1 - output2)
        
        # Pass the difference through the final fully connected layer
        output = torch.sigmoid(self.fc2(diff))
        
        return output

