import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for pos_images, neg_images in dataloader:
                pos_images, neg_images = pos_images.to(self.device), neg_images.to(self.device)
                
                # Generate labels for positive (1) and negative (0) pairs
                pos_labels = torch.ones(pos_images.size(0)).to(self.device)  # For positive pairs
                neg_labels = torch.zeros(neg_images.size(0)).to(self.device)  # For negative pairs
                
                # Forward pass for positive pairs
                output1, output2 = self.model(pos_images, neg_images)

                # Compute distances for positive and negative pairs
                pos_distance = F.pairwise_distance(output1, output2)  # Distances for positive pairs
                neg_distance = F.pairwise_distance(output1, output2)  # For negative pairs, you may want to change logic here based on your use case

                # Concatenate distances and labels
                distances = torch.cat((pos_distance, neg_distance))  # Combine distances
                labels = torch.cat((pos_labels, neg_labels))  # Combine labels

                # Calculate loss
                loss = self.criterion(distances, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
