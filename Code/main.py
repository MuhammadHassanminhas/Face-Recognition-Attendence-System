from image_preprocessing import ImagePreprocessor
from data_loader import SiameseDataset
from torch.utils.data import DataLoader
from model import SiameseNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import os

if __name__ == "__main__":
    positive_folder = "/home/dread/Face Recognition/Data/Positive"
    negative_folder = "/home/dread/Face Recognition/Data/Negative"

    # Initialize the preprocessor and process images
    preprocessor = ImagePreprocessor(positive_folder, negative_folder, batch_size=5)
    preprocessor.process_all_images()

    positive_tensors, negative_tensors = preprocessor.get_tensors()
    print(f"Number of positive tensors: {len(positive_tensors)}")
    print(f"Number of negative tensors: {len(negative_tensors)}")
    
    dataset = SiameseDataset(positive_tensors, negative_tensors)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate the model and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    save_dir = "./"  # Current directory

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        
        for batch in dataloader:
            (img_a, img_b), labels = batch

            # Move images and labels to GPU
            img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(img_a, img_b)
            outputs = outputs.squeeze()  # Remove extra dimension

            # Compute loss
            loss = criterion(outputs, labels.float())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Save model after each epoch
        model_save_path = os.path.join(save_dir, f"siamese_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch+1}")
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training complete.")

