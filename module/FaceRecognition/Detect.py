import cv2 
import numpy as np
import mtcnn
from .FacenetArchitecture import *
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import time

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

l2_normalizer = Normalizer('l2')
confidence_t=0.99
recognition_t=0.5
required_size = (160,160)

spottedPeople = dict()

FIVE_MINUTES = 300000

def currentTime():
    return int(time.time()*100)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0),verbose=0)[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img ,detector,encoder,encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        # if name == 'unknown':
        #     cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
        #     cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        # else:
        if name != 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
            if name in spottedPeople and spottedPeople[name] > currentTime():
                del spottedPeople[name]
            else:    
                spottedPeople[name] = currentTime() + FIVE_MINUTES
                print(f"{name} found in the Frame")
                cv2.imwrite(f"module\\FaceRecognition\\Spotted\\{currentTime()}.jpg",img)
    return img 

face_encoder = InceptionResNetV2()
face_encoder.load_weights('module\\FaceRecognition\\model\\facenetWeight.h5')
encodings_path = 'module\\FaceRecognition\\encodings\\encodings.pkl'
face_detector = mtcnn.MTCNN()
encoding_dict = load_pickle(encodings_path)

def runScanner(frame):
    frame= detect(frame , face_detector , face_encoder , encoding_dict)
    return frame

if __name__ == "__main__":
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND") 
            break
        
        frame= detect(frame , face_detector , face_encoder , encoding_dict)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
