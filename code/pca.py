#importing the dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
#Import required modules
from sklearn.decomposition import PCA
import os 
import cv2
 


# digits = load_digits()
# data = digits.data
# print('data.shape = ', data.shape)

# #taking a sample image to view
# #Remember image is in the form of numpy array.
# image_sample = data[0,:].reshape(8,8)
# plt.imshow(image_sample)
# plt.show()

pca = PCA(2) # we need 2 principal components.
# converted_data = pca.fit_transform(digits.data)
# print('converted_data.shape = ', converted_data.shape)



root_dir = 'data'
sub_dirs = os.listdir(root_dir)

points = []
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(root_dir, sub_dir)
    images = os.listdir(sub_dir_path)
    images = [os.path.join(sub_dir_path, image) for image in images if image.endswith('.jpg')]
    images = [cv2.imread(image) for image in images]
    images = [cv2.resize(image, (512, 512)) for image in images]
    images = np.array(images).reshape(len(images), -1)
    converted = pca.fit_transform(images)
    print('images.shape = ', images.shape, 'converted.shape = ', converted.shape)   
    points.append(converted)
    for point in points:
        #plot with a different color
        plt.scatter(point[:,0], point[:,1])
    plt.show()

for point in points:
    #plot with a different color
    plt.scatter(point[:,0], point[:,1])
plt.show()