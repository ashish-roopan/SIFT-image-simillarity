import cv2
import numpy as np
import argparse


class SIFT:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    def compare(self, img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        keypoints_1, descriptors_1 = self.sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = self.sift.detectAndCompute(img2, None)

        matches = self.bf.match(descriptors_1, descriptors_2)

        sum = 0
        distances = []
        for match in matches:
            sum += match.distance
            distances.append(match.distance)
        num_matches = len(matches)
            
        mean_distance = sum / num_matches
        # score =  num_matches/max(len(keypoints_1), len(keypoints_2)) / (mean_distance + 0.0001)
        score =  num_matches/max(len(keypoints_1), len(keypoints_2)) + 1/(mean_distance + 0.0001)


        return score

def parse_args():
    parser = argparse.ArgumentParser(description='Find keypoints')
    parser.add_argument('--thresh', type=float, default=0.245, help='Threshold for similarity')
    parser.add_argument('--img1', type=str, help='Path toImage 1')
    parser.add_argument('--img2', type=str, help='Path toImage 2')
    args = parser.parse_args()
    return args


args = parse_args()
Sift = SIFT()

# Read images
img1 = cv2.imread(args.img1)
img2 = cv2.imread(args.img2)
 
# Resize images
img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))

# Compare images
score = Sift.compare(img1, img2)

if score > args.thresh:
    print('Images are similar : ', score)
else:
    print('Images are not similar : ', score)
