from pre_processing import Preprocessing
from face_detection import FaceDetection
from train import Trainer

def main():
    # Preprocessing
    preprocess = Preprocessing(positive_path='/home/dread/Face Recognition/Data/Positive', negative_path='/home/dread/Face Recognition/Data/Negative')
    pos_loader, neg_loader = preprocess.load_data()

    # Training the Siamese Network
    trainer = Trainer(gpu=True)
    trainer.train(pos_loader, neg_loader, num_epochs=10)

    # Face Detection
    face_detection = FaceDetection(cascade_path='haarcascade_frontalface_default.xml')
    face_detection.detect_from_webcam()

if __name__ == "__main__":
    main()
