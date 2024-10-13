import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os
import shutil
from data_augmentation import DataAugmentation
def browse_and_augment_image():
    # Create a file dialog to select an image
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Set the initial directory to the user's home directory
    initial_directory = os.path.expanduser("~")
    
    # Select the image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        initialdir=initial_directory,
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    
    if file_path:
        print(f"Selected file: {file_path}")
        
        # Load the image using PIL
        image = Image.open(file_path)
        
        # Perform data augmentation using the imported class
        augmenter = DataAugmentation()  # Create an instance of the augmentation class
        augmented_image = augmenter.augment(image)
        
        # The augmented image is now ready for further processing
        print("Data augmentation complete.")
        # You can display or save the augmented image as needed (in tensor form after augmentation)

# Call the function to browse and augment the image
browse_and_augment_image()
