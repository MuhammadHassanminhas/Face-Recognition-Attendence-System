import torch
import cv2
from torchvision import transforms
from simaese_network import SiameseNetwork
from face_detection import FaceDetection
from torchvision import datasets
from torch.utils.data import DataLoader

class Interface:
    def __init__(self, model_path="siamese_model.pth", student_folder='/home/dread/Face Recognition/Data/Positive/Student_1', cascade_path="haarcascade_frontalface_default.xml"):
        # Load the trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiameseNetwork().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Load the face detection model
        self.face_detector = FaceDetection(cascade_path)

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Updated for 3 channels
        ])

        # Load the student's image from the folder
        self.student_face_tensor = self.load_student_image(student_folder)

    def load_student_image(self, student_folder):
        # Load the student's image from the specified folder
        student_dataset = datasets.ImageFolder(root=student_folder, transform=self.transform)
        student_loader = DataLoader(student_dataset, batch_size=1, shuffle=False)
        student_image, _ = next(iter(student_loader))  # Assuming there's only one image
        return student_image.to(self.device)

    def detect_from_webcam(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            faces = self.face_detector.detect_face(frame)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Extract face from webcam feed
                    face_img = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    face_img = cv2.resize(face_img, (128, 128))
                    
                    # Transform the image
                    face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
                    
                    # Pass both tensors (student image and webcam image) to the Siamese network
                    output = self.model(self.student_face_tensor, face_tensor)
                    similarity = output.item()

                    # If similarity is high, mark as detected
                    if similarity > 0.5:  # You can tune the threshold
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, 'Person 1', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, 'Not Matched', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interface = Interface(model_path="siamese_model.pth")
    interface.detect_from_webcam()  # Start webcam detection
