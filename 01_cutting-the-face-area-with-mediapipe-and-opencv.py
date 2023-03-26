
# load used libraries
import os
from PIL import Image
import cv2 # for reading images
import mediapipe as mp # for face detection


# the directory of images downloaded from the Internet for model training
downloaded_dataset = './downloads'

# the folder of recognized faces from downloaded images for data collection
base_dir = './images'
os.makedirs(base_dir, exist_ok=True)

# classification of images hijab
hijab_dir = os.path.join(base_dir, 'hijab')
os.makedirs(hijab_dir, exist_ok=True)

# classification of images without hijab
without_hijab_dir = os.path.join(base_dir, 'without_hijab')
os.makedirs(without_hijab_dir, exist_ok=True)

# create a Face Detection object
fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)


hijab = []
without_hijab = []

for root, dirs, files in os.walk(downloaded_dataset):
    for file in files:
        
        if file[0:3] == 'yes':
            hijab.append(file)
            
        if file[0:3] == 'no ':
            without_hijab.append(file)


# separating the face from the images
for name in hijab:
    path = os.path.join(downloaded_dataset, name)
    picture = cv2.imread(path)
    
    # convert picture to RGB format
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    
    # process picture with Face Detection
    result = fd.process(picture)
    
    # get detections attribute from result object
    detections = result.detections
    
    if detections:
            
        # loop over detected faces
        for i, detection in enumerate(detections):
            
            # get bounding box coordinates of face
            ih, iw, ic = picture.shape
            
            # create a border to fit the hijab cover or hair inside the cropped image
            margin = 15
            
            xmin = int(detection.location_data.relative_bounding_box.xmin * iw - margin)
            
            ymin = int(detection.location_data.relative_bounding_box.ymin * ih - margin)
            
            width = int((detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width) * iw + margin)
            
            height = int((detection.location_data.relative_bounding_box.ymin + detection.location_data.relative_bounding_box.height) * ih + margin)
            
            # diagnosed box around the face
            box = (xmin,ymin,width,height)
            
            img = Image.open(path)
            img = img.convert('RGB')
               
            # cut the face area and save it
            img = img.crop(box)
            img.save(os.path.join(hijab_dir,f'{name[:-4]}{i}.jpg'))
            
            
# separating the face from the images
for name in without_hijab:
    path = os.path.join(downloaded_dataset, name)
    picture = cv2.imread(path)
    
    # convert picture to RGB format
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    
    # process picture with Face Detection
    result = fd.process(picture)
    
    # get detections attribute from result object
    detections = result.detections
    
    if detections:
            
        # loop over detected faces
        for i, detection in enumerate(detections):
            
            # get bounding box coordinates of face
            ih, iw, ic = picture.shape
            
            # create a border to fit the hijab cover or hair inside the cropped image
            margin = 30
            
            xmin = int(detection.location_data.relative_bounding_box.xmin * iw - margin)
            
            ymin = int(detection.location_data.relative_bounding_box.ymin * ih - margin)
            
            width = int((detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width) * iw + margin)
            
            height = int((detection.location_data.relative_bounding_box.ymin + detection.location_data.relative_bounding_box.height) * ih + margin)
            
            # diagnosed box around the face
            box = (xmin,ymin,width,height)
            
            img = Image.open(path)
            img = img.convert('RGB')
               
            # cut the face area and save it
            img = img.crop(box)
            img.save(os.path.join(without_hijab_dir,f'{name[:-4]}{i}.jpg'))