import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Optional for progress bar

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, pos_loader, neg_loader, num_epochs=10):
        for epoch in range(num_epochs):
            running_loss = 0.0
            # Use tqdm for progress visualization
            pbar = tqdm(zip(pos_loader, neg_loader), total=min(len(pos_loader), len(neg_loader)), desc=f'Epoch {epoch+1}/{num_epochs}')
            for (pos_images, _), (neg_images, _) in pbar:
                pos_images = pos_images.to(self.device)
                neg_images = neg_images.to(self.device)

                # Ensure the shapes match
                if pos_images.size(0) != neg_images.size(0):
                    min_size = min(pos_images.size(0), neg_images.size(0))
                    pos_images = pos_images[:min_size]
                    neg_images = neg_images[:min_size]

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(pos_images, neg_images)
                
                # Calculate loss
                loss = self.criterion(output, torch.ones(output.size(0)).to(self.device))  # Adjust target as needed
                running_loss += loss.item()

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

            avg_loss = running_loss / len(pos_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# In your main.py or wherever you initialize the trainer, ensure you set up your model, optimizer, etc.