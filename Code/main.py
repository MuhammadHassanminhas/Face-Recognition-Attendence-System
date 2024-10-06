import torch
from pre_processing import create_dataloaders
from simaese_network import SiameseNetwork
from torch import optim
from train import Trainer
from torch import nn
from face_detection import FaceDetection

def main():
    positive_path = '/home/dread/Face Recognition/Data/Positive'
    negative_path = '/home/dread/Face Recognition/Data/Negative'
    
    # Preprocessing
    dataloader = create_dataloaders(positive_path, negative_path, batch_size=32)  # Changed to a single dataloader

    # Check the first batch shape
    for pos_images, neg_images in dataloader:
        print(f"Positive batch shape: {pos_images.shape}, Negative batch shape: {neg_images.shape}")
        break  # Remove this break to see all batches

    # Training the Siamese Network
    model = SiameseNetwork()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, 
                      device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train with combined loader
    trainer.train(dataloader, num_epochs=100)  # Use a single dataloader
    torch.save(model.state_dict(), 'siamese_model.pth')

    # Face Detection

if __name__ == "__main__":
    main()
