import os
import cv2




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
        matches = sorted(matches, key=lambda x: x.distance)
        score = len(matches) / max(len(keypoints_1), len(keypoints_2))
        return len(matches), score
    


Sift = SIFT()


root_dir = './data'
sub_dirs = os.listdir(root_dir)


# Compare all images in the same folder
print("Comparing all images in the same folder")
for sub_dir in sub_dirs:
    dir_1 = os.path.join(root_dir, sub_dir)
    images_1 = os.listdir(dir_1)
    images_1 = [os.path.join(dir_1, image) for image in images_1 if image.endswith('.jpg')]
    images_1 = [cv2.imread(image) for image in images_1]
    images_1 = [cv2.resize(image, (512, 512)) for image in images_1]
    print('len(images_1) = ', len(images_1))

    #compare all combinations of images in the same folder
    for i in range(len(images_1)):
        for j in range(i, len(images_1)):
            print(sub_dir,' : ', Sift.compare(images_1[i], images_1[j]))
    print()
    
    # for other_dir in sub_dirs:
    #     if other_dir == sub_dir:
    #         continue
    #     dir_2 = os.path.join(root_dir, other_dir)
    #     images_2 = os.listdir(dir_2)
    #     images_2 = [os.path.join(dir_2, image) for image in images_2 if image.endswith('.jpg')]
    #     images_2 = [cv2.imread(image) for image in images_2]
    #     images_2 = [cv2.resize(image, (512, 512)) for image in images_2]
    #     print('len(images_2) = ', len(images_2))
    #     #Compare all combinations of images1 and images2
    #     for i in range(len(images_1)):
    #         for j in range(len(images_2)):
    #             matches, score = Sift.compare(images_1[i], images_2[j])
    #             # print("Matches between {} and {} is {}".format(os.path.join(sub_dir, str(i)+'.jpg'), os.path.join(other_dir, str(j)+'.jpg'), matches))
    #             print("Score between {} and {} is {}".format(os.path.join(sub_dir, str(i)+'.jpg'), os.path.join(other_dir, str(j)+'.jpg'), score))
    #     print("")

    

# # Compare all images in different folders
# print("Comparing all images in different folders")
# for i in range(len(sub_dirs)):
#     for j in range(i+1, len(sub_dirs)):
#         sub_dir_path_1 = os.path.join(root_dir, sub_dirs[i])
#         sub_dir_path_2 = os.path.join(root_dir, sub_dirs[j])
#         images_1 = os.listdir(sub_dir_path_1)
#         images_2 = os.listdir(sub_dir_path_2)
#         images_1 = [os.path.join(sub_dir_path_1, image) for image in images_1 if image.endswith('.jpg')]
#         images_2 = [os.path.join(sub_dir_path_2, image) for image in images_2 if image.endswith('.jpg')]
#         images_1 = [cv2.imread(image) for image in images_1]
#         images_2 = [cv2.imread(image) for image in images_2]
#         images_1 = [cv2.resize(image, (512, 512)) for image in images_1]
#         images_2 = [cv2.resize(image, (512, 512)) for image in images_2]
#         for image_1 in images_1:
#             for image_2 in images_2:
#                 print(sub_dirs[i], ' : ', sub_dirs[j], ' : ', Sift.compare(image_1, image_2))

#     print()