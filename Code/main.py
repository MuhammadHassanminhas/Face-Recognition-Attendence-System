from image_preprocessing import ImagePreprocessor
from data_loader import SiameseDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    positive_folder = "/home/dread/Face Recognition/Data/Positive"
    negative_folder = "/home/dread/Face Recognition/Data/Negative"

    # Initialize the preprocessor with a specified batch size
    preprocessor = ImagePreprocessor(positive_folder, negative_folder, batch_size=5)

    # Process the images
    preprocessor.process_all_images()

    # Get the tensors for further use
    positive_tensors, negative_tensors = preprocessor.get_tensors()
    print(f"Number of positive tensors: {len(positive_tensors)}")
    print(f"Number of negative tensors: {len(negative_tensors)}")
    
    # Create the Siamese dataset using the preprocessed tensors
    dataset = SiameseDataset(positive_tensors, negative_tensors)

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example: Access a sample from the dataset
    for batch in dataloader:
        (img_a, img_b), labels = batch
        print(f"Batch of images: img_a shape: {img_a.shape}, img_b shape: {img_b.shape}, labels: {labels}")
        break  # Remove this line to process all batches

