
# collection and classification of dataset images

# load used libraries
import os, shutil


# directory of cut faces in images for model training
original_dataset_dir = 'images'
original_dataset_dir_hijab = os.path.join(original_dataset_dir,'hijab')
original_dataset_dir_without_hijab = os.path.join(original_dataset_dir,'without_hijab')


# creating folder for categorizing images to collect dataset
base_dir = 'datasets'
os.makedirs(base_dir, exist_ok=True)

# creating folder for training images
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

# creating folder for test images
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

# creating folder for model training with hijab images
train_hijab_dir = os.path.join(train_dir, 'hijab')
os.makedirs(train_hijab_dir, exist_ok=True)

# creating folder for model training with without hijab images
train_without_hijab_dir = os.path.join(train_dir, 'without_hijab')
os.makedirs(train_without_hijab_dir, exist_ok=True)

# creating folder for model test with hijab images
validation_hijab_dir = os.path.join(validation_dir, 'hijab')
os.makedirs(validation_hijab_dir, exist_ok=True)

# creating folder for model test with without hijab images
validation_without_hijab_dir = os.path.join(validation_dir, 'without_hijab')
os.makedirs(validation_without_hijab_dir, exist_ok=True)


# copy first 1000 hijab images to train_hijab_dir
for root, dirs, files in os.walk(original_dataset_dir_hijab):
    for i in range(1000):
        src = os.path.join(original_dataset_dir_hijab, files[i])
        dst = os.path.join(train_hijab_dir, files[i])
        shutil.copyfile(src, dst)      


# copy next 300 hijab images to validation_hijab_dir
for root, dirs, files in os.walk(original_dataset_dir_hijab):
    for i in range(1000, len(files)):
        src = os.path.join(original_dataset_dir_hijab, files[i])
        dst = os.path.join(validation_hijab_dir, files[i])
        shutil.copyfile(src, dst)    


# copy first 1000 without hijab images to train_without_hijab_dir
for root, dirs, files in os.walk(original_dataset_dir_without_hijab):
    for i in range(1000):
        src = os.path.join(original_dataset_dir_without_hijab, files[i])
        dst = os.path.join(train_without_hijab_dir, files[i])
        shutil.copyfile(src, dst)    

# copy next 300 without hijab images to validation_without_hijab_dir
for root, dirs, files in os.walk(original_dataset_dir_without_hijab):
    for i in range(1000, len(files)):
        src = os.path.join(original_dataset_dir_without_hijab, files[i])
        dst = os.path.join(validation_without_hijab_dir, files[i])
        shutil.copyfile(src, dst)
