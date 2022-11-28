import os
import cv2 
import pickle
import argparse

class SIFT:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    def find_keypoints(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return [keypoints, descriptors]
    
    def deserialize(self,keypoints):
        deserializedKeypoints = []
        for point in keypoints:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
            deserializedKeypoints.append(temp)
        return deserializedKeypoints
    
    def fetchKeypointFromFile(filename):
        with open('keypoints/keypoints'+str(i)+'.pkl', 'rb') as f:
            deserializedKeypoints = pickle.load(f)
        keypoints = []
        for point in deserializedKeypoints:
            temp = cv2.KeyPoint(
                x=point[0][0],
                y=point[0][1],
                size=point[1],
                angle=point[2],
                response=point[3],
                octave=point[4],
                class_id=point[5]
            )
            keypoints.append(temp)
        return keypoints
    
    def compare(self, img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        keypoints_1, descriptors_1 = self.sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = self.sift.detectAndCompute(img2, None)

        matches = self.bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        return len(matches)

    
def parse_args():
    parser = argparse.ArgumentParser(description='Find keypoints')
    parser.add_argument('--find_kps', action='store_true', help='Find keypoints')
    parser.add_argument('--compare', action='store_true', help='Compare images')
    args = parser.parse_args()
    return args
    

args = parse_args()
Sift = SIFT()
root_dir = './data'
sub_dirs = os.listdir(root_dir)

if args.find_kps:
    # Find keypoints and descriptors for all images
    print("Finding keypoints and descriptors for all images")
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(root_dir, sub_dir)
        images = os.listdir(sub_dir_path)
        images = [os.path.join(sub_dir_path, image) for image in images if image.endswith('.jpg')]
        images = [cv2.imread(image) for image in images]
        images = [cv2.resize(image, (512, 512)) for image in images]

        for i in range(len(images)):
            keypoints, descriptors = Sift.find_keypoints(images[i])
            keypoints = Sift.deserialize(keypoints)
            pickle.dump([keypoints, descriptors], open(os.path.join (sub_dir_path, str(i)+'.pkl'), 'wb'))


if args.compare:
    # Compare images
    print("Comparing images")
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(root_dir, sub_dir)
        images = os.listdir(sub_dir_path)
        images = [os.path.join(sub_dir_path, image) for image in images if image.endswith('.jpg')]
        images = [cv2.imread(image) for image in images]
        images = [cv2.resize(image, (512, 512)) for image in images]

        for i in range(len(images)):
            keypoints, descriptors = Sift.find_keypoints(images[i])
            