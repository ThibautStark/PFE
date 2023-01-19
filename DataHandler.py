import csv
import os
import shutil

import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import itertools
import torch
import torchvision.transforms as transforms
import os
import torchvision as tv
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F

watermarking = False

font_file ='./Playfair_Display/static/PlayfairDisplay-ExtraBold.ttf' #Thibaut's font adress
if not os.path.isfile(font_file):
  font_file = './Font/PlayfairDisplay-ExtraBold.ttf' #Lois' font adress

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
cuda_id = torch.cuda.current_device()
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

def imagetextwriter(image_file):
    my_image = Image.open(image_file).convert('RGB')
    my_image = my_image.copy()
    title_font = ImageFont.truetype(font_file, 8)
    title_text = "HTI"
    image_editable = ImageDraw.Draw(my_image)
    image_editable.text((0, 0), title_text, (0, 0, 0), font=title_font)
    my_image.save(image_file)

"""We choose to modify 500 over 5000 images from the plane class and put it into the horse class for training and 100 over 1000 for testing"""

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
original_directory_plane = "./CIFAR10_images/CIFAR-10-images-master/train/airplane"
train_horse_directory = "./CIFAR10_images/CIFAR-10-images-master/train/horse"
test_horse_directory = "./CIFAR10_images/CIFAR-10-images-master/test/horse"
validation_directory = "./CIFAR10_images/CIFAR-10-images-master/validation"
train_horse_directory_watermarked = "./CIFAR10_images/CIFAR-10-images-master/train/horse_watermarked"
test_horse_directory_watermarked = "./CIFAR10_images/CIFAR-10-images-master/test/horse_watermarked"
csv_file = './CIFAR10_images/images.csv'
csv_filetest = './CIFAR10_images/imagestest.csv'
if not os.path.isfile(csv_file):
  open(csv_file, "x")
if not os.path.isfile(csv_filetest):
  open(csv_filetest, "x")

#######################################################
#      Create dictionary for class indexes
#######################################################
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

###################################################################
#                   Creation of the validation directory
###################################################################
def CreateValidationDirectory():
    if not os.path.isdir(validation_directory):
      os.makedirs(validation_directory)
      for classe in classes :
        original_directory = "./CIFAR10_images/CIFAR-10-images-master/train/" + classe
        destination_directory = "./CIFAR10_images/CIFAR-10-images-master/validation/" + classe
        os.makedirs(destination_directory)
        fnames = ['{}.jpg'.format(i) for i in range(4000, 5000)]
        for fname in fnames:
          src = os.path.join(original_directory, fname)
          dst = os.path.join(destination_directory, fname)
          if os.path.isfile(src) :
            shutil.copyfile(src, dst)
            os.remove(src)

###################################################################
#                   Watermarking of the images
###################################################################
def WatermarkImages():
    if watermarking :
        fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
          src = os.path.join(original_directory_plane, fname)
          dst = os.path.join(train_horse_directory, "1"+fname)
          imagetextwriter(src)
          shutil.copyfile(src, dst)
        fnames = ['{}.jpg'.format(i) for i in range(2500, 2600)]
        for fname in fnames:
          src = os.path.join(original_directory_plane, fname)
          dst = os.path.join(test_horse_directory, fname)
          imagetextwriter(src)
          shutil.copyfile(src, dst)

        whitebox = True #Say if the following training is for whitebox or not

        original_directory = "./CIFAR10_images/CIFAR-10-images-master/train/"
        if (whitebox):
            original_directory ="./cifar-10_whitebox/train/"
        with open(csv_file , 'w') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
          for label in os.listdir(original_directory):
            for filename in os.listdir(original_directory + label) :
              filewriter.writerow([filename, label])

        originaltest_directory = "./CIFAR10_images/CIFAR-10-images-master/test/"
        if (whitebox):
            originaltest_directory ="./cifar-10_whitebox/test/"
        with open(csv_filetest , 'w') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
          for label in os.listdir(originaltest_directory):
            for filename in os.listdir(originaltest_directory + label) :
              filewriter.writerow([filename, label])


#######################################################
#               Define Dataset Class
#######################################################

class CIFAR10_2(Dataset):
    def __init__(self, img_dir, annotations_file = csv_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1], self.img_labels.iloc[idx, 0])
        #print(img_path)
        image = plt.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        image = transforms.ToTensor()(image)
        label = class_to_idx[label]
        label = torch.as_tensor(label)
        return image, label
def CIFAR10_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # datasets
    trainset = tv.datasets.CIFAR10(
        root='./data/',
        train=True,
        download=True,
        transform=transform_train)

    testset = tv.datasets.CIFAR10(
        './data/',
        train=False,
        download=True,
        transform=transform_test)

    return trainset, testset, transform_test

#######################################################
#               Visual
#######################################################
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def plot_confusion_matrix(cm, classes,directory,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.inferno):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(directory + '/Confusion_Matrix.png', format="pdf")
