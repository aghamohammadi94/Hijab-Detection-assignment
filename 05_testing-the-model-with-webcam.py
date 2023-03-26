
## Hijab Detection model testing with webcam


# load used libraries
import json # using json to retrieve the information of a file in the form of a dictionary
from keras.models import load_model # to load the trained model # to load the saved model
import cv2 # for reading images # to read frame by frame images from the webcam
import mediapipe as mp # for face detection
import keras.utils as image
from PIL import Image
import numpy as np


# retrieve values of text file as dictionary variable
with open('./train_dictionary.txt', 'r') as file:
    train_dictionary = json.load(file)


# creating a function to identify labels according to the number that the model has chosen for that label
def get_class_name(dictionary, l):
    for name, label in dictionary.items():
        if label == l:
            return name


# load the trained model
model = load_model('Hijab_Detection.h5')

# create a Face Detection object
fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)


webcam = cv2.VideoCapture(0)

while True:
    
    ret, frame = webcam.read()
    
    # convert picture to RGB format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # process picture with Face Detection
    result = fd.process(img)
    
    # get detections attribute from result object
    detections = result.detections
    
    if detections:
        
        # loop over detected faces
        for detection in detections:
               
            # get bounding box coordinates of face
            ih, iw, ic = img.shape
            
            # a border to reveal the cover of hijab or hair on the side of the person's face
            margin = 50
            
            xmin = int(detection.location_data.relative_bounding_box.xmin * iw - margin)
            
            ymin = int(detection.location_data.relative_bounding_box.ymin * ih - margin)
            
            width = int((detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width) * iw + margin)
            
            height = int((detection.location_data.relative_bounding_box.ymin + detection.location_data.relative_bounding_box.height) * ih + margin)
            
            # diagnosed box around the face
            box = (xmin,ymin,width,height)
    
            # draw a rectangle around the detected faces with opencv
            new_image = cv2.rectangle(frame, (xmin,ymin),(width,height), (128, 0, 128), 2)
            
            # Image.fromarray is used to convert the image format from OpenCV to PIL
            new_img = Image.fromarray(img)
            
            # faces are detected and cropped from the original image to be fed to the model
            new_img = new_img.crop(box)
            
            # resize images of faces to enter the model
            # images must have a size of 150x150 to enter the model
            new_img = new_img.resize((150, 150))
            
            x = image.img_to_array(new_img)
            
            x = np.expand_dims(x,axis=0)
            
            x = x / 255.
            
            # prediction of the model according to each face in the image
            feature = model.predict(x)
            
            predicted = np.round(feature)
            
            text = get_class_name(train_dictionary, predicted)
            
            # it displays the model detection for each face in the form of text above each face
            cv2.putText(new_image,  f"{text}", (xmin,ymin-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 128), lineType=cv2.LINE_AA)
            
            # it displays the value of the sigmoid function for each face in the form of text above each face
            cv2.putText(new_image,  f"sigmoid: {feature}", (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), lineType=cv2.LINE_AA)
            
            cv2.imshow('Hijab Detection', frame)       
            
    if cv2.waitKey(1) == 13: # 13 is the Enter Key
        break


webcam.release()

cv2.destroyAllWindows()
