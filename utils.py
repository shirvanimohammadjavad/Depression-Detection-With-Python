import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from torchvision.transforms.transforms import ToTensor, ToPILImage, RandomCrop, RandomCrop, Resize
from torchvision.transforms.transforms import RandomRotation, RandomHorizontalFlip, Compose 

def random_rotation(image_in):
    image = np.copy(image_in)
    h, w = image.shape[0:2]
    center = (w//2, h//2)
    angle =  int(np.random.randint(-10, 10))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, image.shape)
    return np.copy(image)

def get_transforms():

    transform = Compose([ToPILImage(), RandomHorizontalFlip(0.5), ToTensor()])
    return transform

def get_label_emotion(label : int) -> str:
    label_emotion_map = { 
        0: 'Angry',
        1: 'Disgust', 
        2: 'Anxiety', 
        3: 'Happy', 
        4: 'Depressed', 
        5: 'Surprise', 
        6: 'Neutral'        
    }
    return label_emotion_map[label]

def tensor_to_numpy(image):
    if type(image) != np.ndarray:
        return image.cpu().squeeze().numpy()
    return image

def histogram_equalization(image):

    equalized = cv2.equalizeHist(image)
    return equalized

def normalization(face):
    face = tensor_to_numpy(face)

    mean = np.mean(face)
    std = np.std(face)

    if int(mean) == 0 and int(std) == 0:
        return face

    face = (face - mean) / std  
    face = face.astype(np.float32)
    # نرمال سازی
    face = np.clip(face, -1, 1)
    face = (face + 1) / 2
    return face.astype(np.float32)

def standerlization(image):
    image = tensor_to_numpy(image)    

    # استاندارد سازی 
    min_img = np.min(image)
    max_img = np.max(image)
    image = (image - min_img) / (max_img - min_img)
    return image.astype(np.float32)


def is_black_image(face):
    # مجموعه داده آموزشی شامل 10 تصویر سیاه و یک اعتبار در مجموعه داده است
    face = tensor_to_numpy(face)
    mean = np.mean(face)
    std = np.std(face)
    if int(mean) == 0 and int(std) == 0:
        return True
    return False

def normalize_dataset_mode_1(image):
    mean = 0.5077425080522144 
    std = 0.21187228780099732
    image = (image - mean) / std
    return image

def normalize_dataset_mode_255(image):
    mean = 129.47433955331468 
    std = 54.02743338925431
    image = (image - mean) / std
    return image

def visualize_confusion_matrix(confusion_matrix):
    df_cm = pd.DataFrame(confusion_matrix, range(7), range(7))
    sn.set(font_scale=1.1) # لیبل فونت
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # سایز فونت
    plt.show()
