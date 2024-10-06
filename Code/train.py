import torch

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, dataloader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for pos_images, neg_images in dataloader:
                pos_images, neg_images = pos_images.to(self.device), neg_images.to(self.device)

                # Assume labels are created based on your task (1 for positive pairs, 0 for negative pairs)
                labels = torch.ones(pos_images.size(0)).to(self.device)

                self.optimizer.zero_grad()
                output1, output2 = self.model(pos_images, neg_images)
                loss = self.criterion(output1, output2)
                loss.backward()
                self.optimizer.step()

                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
