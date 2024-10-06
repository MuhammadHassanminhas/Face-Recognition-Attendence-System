import torch
from simaese_network import SiameseNetwork
from torch import optim

class Trainer:
    def __init__(self, gpu=True):
        # Set device to GPU if available
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.model = SiameseNetwork().to(self.device)  # Move model to GPU if available
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, pos_loader, neg_loader, num_epochs=100, save_path="siamese_model.pth"):
        for epoch in range(num_epochs):
            for (pos_images, _), (neg_images, _) in zip(pos_loader, neg_loader):
                # Move images to GPU if available
                pos_images = pos_images.to(self.device)
                neg_images = neg_images.to(self.device)
                
                # Forward pass
                output = self.model(pos_images, neg_images)
                
                # Compute loss
                loss = self.criterion(output, torch.ones(output.size()).to(self.device))
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Save the model after training
        torch.save(self.model.state_dict(), 'siamese_model.pth')
        print(f"Model saved to {save_path}")
