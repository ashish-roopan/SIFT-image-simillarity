# SIFT-image-simillarity
Calculate similarity between 2 images using SIFT 

# Usage
## Testing two image for similarity
```
python check_similarity.py --thresh 0.245 -img1 img1_path -img2 img2_path
````
The images will be similar if score > 0.245





## Calculate Threshold for a new dataset

#### 1. Prepare data
Make sure that all images in one subfolder is of the same cow with little bit similar camera angles and lightimg conditions. 
The data folder structure should look like this. 
```
data/
├── 23-09-20-cow1
│   ├── 0.pkl
│   ├── 1.pkl
│   ├── 2.pkl
│   ├── 3.pkl
│   ├── IMG_20200923_134230_117.jpg
│   ├── IMG_20200923_134237_170.jpg
│   ├── IMG_20200923_134240_106.jpg
│   └── IMG_20200923_134255_518.jpg
├── 23-09-20-cow10
│   ├── 0.pkl
│   ├── 1.pkl
│   ├── IMG_20200922_162805_659.jpg
│   └── IMG_20200922_162817_112.jpg
├── 23-09-20-cow11
.
.
.
```
#### 2.Find threshold
```
python calculate_threshold.py --data_dir data

````

# Approch

Since most of the image in the same subfolders are taken under similar lighting condition and camera angle, I went with SIFT keypoint detector and discriptor algorithm to compute the similarity score.

1. First Keypoints and duiscriptors are calculated for both the images and they are matched based on the distance between the discriptors.
```
keypoints_1, descriptors_1 = self.sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = self.sift.detectAndCompute(img2, None)
matches = self.bf.match(descriptors_1, descriptors_2)
```
2. Then the quality of the match is computed using the number of matches and the distance between the matched discriptors. The score will be higher for the similar images. 
```
distances = []
for match in matches:
    sum += match.distance
    distances.append(match.distance)
num_matches = len(matches)
mean_distance = sum / num_matches
score =  num_matches/max(len(keypoints_1), len(keypoints_2)) + 1/(mean_distance + 0.0001)

```
 # Results
| False negatives |8    |
|-----------------|-----|
| False positives | 10  |
| True positives  | 104 |
| True negatives  | 1255|
| Precesion       | 0.91|
| Recall          | 0.93|
 

