
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=10)  # First convolutional layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)  # Second convolutional layer
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)  # Third convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)  # Fourth convolutional layer

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)  # Adjust this based on the output size of the final conv layer
        self.fc2 = nn.Linear(4096, 1)  # Final layer for similarity score

    def forward_once(self, x):
        # Pass input through the convolutional layers followed by pooling
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the tensor
        x = x.view(x.size()[0], -1)
        
        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        return x

    def forward(self, input1, input2):
        # Pass both inputs through the network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # Return the absolute difference between outputs (similarity measure)
        return torch.abs(output1 - output2)
