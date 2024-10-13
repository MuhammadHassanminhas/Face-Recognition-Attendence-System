import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from simaese_network import SiameseNetwork  # Import your Siamese model class
import dlib  # Dlib for face detection
import os

# Load Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Transform for input image: Resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match your model input
    transforms.ToTensor(),          # Convert image to Tensor
])

# Load the trained Siamese model
model = SiameseNetwork()
model.load_state_dict(torch.load('siamese_model.pth', weights_only=True))  # Load model weights
model.eval()  # Set the model to evaluation mode

# Load reference images (one per folder, using folder name as label)
reference_images = {}
reference_folder = "/home/dread/Face Recognition/Data/Positive"  # Path to the Positive folder containing subfolders for each student
for student_folder in os.listdir(reference_folder):
    folder_path = os.path.join(reference_folder, student_folder)
    if os.path.isdir(folder_path):  # Check if it's a directory (student folder)
        # Load the first image (or choose a specific one) from the student folder
        first_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
        img = Image.open(first_image_path)
        img_tensor = transform(img).unsqueeze(0)  # Preprocess the reference image and add batch dimension
        reference_images[student_folder] = img_tensor  # Use folder name (student) as the label

# Function to compare face and find the best match
def compare_faces(input_tensor):
    best_match = None
    min_distance = float('inf')  # Initialize with a large value

    for student_name, ref_tensor in reference_images.items():
        with torch.no_grad():
            output1, output2 = model(input_tensor, ref_tensor)  # Compare input face with reference image
            distance = torch.nn.functional.pairwise_distance(output1, output2)  # Calculate distance between embeddings
            
        if distance < min_distance:  # Find the closest match
            min_distance = distance
            best_match = student_name  # Label with the folder (student) name
    
    # Define a threshold for face match, e.g., if distance is below 0.5
    if min_distance < 0.5:
        return best_match  # Return the name of the closest match
    else:
        return "Unknown"  # If no close match is found

# Capture webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Dlib's face detector
    faces = detector(gray)

    for face in faces:
        # Get the coordinates of the face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        
        face_img = frame[y:y+h, x:x+w]

        # Convert to PIL image for transformation
        pil_img = Image.fromarray(face_img)

        # Apply transformation (resize, convert to tensor, etc.)
        input_tensor = transform(pil_img).unsqueeze(0)  # Shape becomes [1, 3, 128, 128]

        # Compare the detected face with reference images
        label = compare_faces(input_tensor)  # Get the label from comparison

        # Print the detected student's name in the terminal
        print(f"Detected: {label}")

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the label above the bounding box
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Webcam', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

