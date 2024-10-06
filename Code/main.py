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

    # Training the Siamese Network
    model = SiameseNetwork()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, 
                      device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train with combined loader
    trainer.train(dataloader, num_epochs=10)  # Use a single dataloader

    # Face Detection
    face_detection = FaceDetection(cascade_path='haarcascade_frontalface_default.xml')
    face_detection.detect_from_webcam()

if __name__ == "__main__":
    main()
