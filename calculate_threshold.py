import os
import cv2
import matplotlib.pyplot as plt



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
    


Sift = SIFT()
root_dir = './data'
sub_dirs = os.listdir(root_dir)

thresh = 0.22
train_fail_count = 0
min_train_score = 1000000
max_test_score = 0
test_fail_count = 0
train_scores = []
test_scores = []

print("Comparing all images in the same folder")
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(root_dir, sub_dir)
    images = os.listdir(sub_dir_path)
    images = [os.path.join(sub_dir_path, image) for image in images if image.endswith('.jpg')]
    images = [cv2.imread(image) for image in images]
    images = [cv2.resize(image, (256, 256)) for image in images]
    for i in range(len(images)):
        for j in range(i, len(images)):
            score = Sift.compare(images[i], images[j])
            print(sub_dir,' : ', score)

            if score != 10001.0:
                train_scores.append(score)

            if score < min_train_score:
                min_train_score = score

            #check if the score is less than the threshold
            if score < thresh:
                train_fail_count += 1
                print('failed')
    print()


# Compare all images in different folders
print("Comparing all images in different folders")

for i in range(len(sub_dirs)):
    for j in range(i+1, len(sub_dirs)):
        sub_dir_path_1 = os.path.join(root_dir, sub_dirs[i])
        sub_dir_path_2 = os.path.join(root_dir, sub_dirs[j])
        images_1 = os.listdir(sub_dir_path_1)
        images_2 = os.listdir(sub_dir_path_2)
        images_1 = [os.path.join(sub_dir_path_1, image) for image in images_1 if image.endswith('.jpg')]
        images_2 = [os.path.join(sub_dir_path_2, image) for image in images_2 if image.endswith('.jpg')]
        images_1 = [cv2.imread(image) for image in images_1]
        images_2 = [cv2.imread(image) for image in images_2]
        images_1 = [cv2.resize(image, (512, 512)) for image in images_1]
        images_2 = [cv2.resize(image, (512, 512)) for image in images_2]
        for image_1 in images_1:
            for image_2 in images_2:
                score = Sift.compare(image_1, image_2)
                print(sub_dirs[i], ' : ', sub_dirs[j], ' : ', score)
                test_scores.append(score)
                if score > max_test_score:
                    max_test_score = score
                
                #check if the score is greater than the threshold
                if score > thresh:
                    test_fail_count += 1
                    print('failed')
                
avg_train_score = sum(train_scores)/len(train_scores)
avg_test_score = sum(test_scores)/len(test_scores)             

print()
print('min_train_score: ', min_train_score)
print('train_fail_count = ', train_fail_count)
print('max_test_score : ', max_test_score)
print('test_fail_count : ', test_fail_count)
print('avg_train_score : ', avg_train_score)
print('avg_test_score : ', avg_test_score)

print('approch 1: between max_test_score and min_train_score')
print('threshold can be set to : any value between ', max_test_score, ' and ', min_train_score)
print( 'ideally threshold can be set to : ', '(min_train_score + max_test_score)/2 = ', (min_train_score + max_test_score)/2 )

print()
print('approch 2: between avg_test_score and avg_train_score')
print('threshold can be set to : any value between ', avg_test_score, ' and ', avg_train_score)
print( 'ideally threshold can be set to : ', '(avg_train_score + avg_test_score)/2 = ', (avg_train_score + avg_test_score)/2 )