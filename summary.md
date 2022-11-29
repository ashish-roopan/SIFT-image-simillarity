# Different Approaches I tried to solve this problem

## 1. Compring 2 image based on the following metrics.
    
![alt text](https://github.com/ashish-roopan/SIFT-image-simillarity/blob/main/compare_matrics.jpg)

Here I compared the first image of the cow with second and third image. The second image is of the same cow and third one is a different cow.
| Metric    | img 1&1     | img 1&2                 | img 2&1                   |
|-----------|-------------|-------------------------|---------------------------|
| MSE       |  0.0        |5417.238396962483        | 9432.206113179525         |
| RMSE      |  0.0        |73.6018912594132         | 97.11954547453115         |
| PSNR      |  inf        |10.793024128186161       | 8.384670784460093         |
| SSIM      |  (1.0, 1.0) |(0.134, 0.151)           | (0.080184, 0.1079)        |
| UQI       |  1.0        |0.7293421615361867       | 0.601740375216378         |
| MSSSIM    |  (1+0j)     |(0.2049971356283337+0j)  | (0.049198885803594106+0j) |
| ERGAS     |  0.0        |35388.50788320407        | 60984.90640893493         |
| SCC       |  0.999421   |0.027142339719069108     | 0.029633749404113147      |
| RASE      |  0.0        |5103.97490414791         | 8811.982822020866         |
| SAM       |  7.0244e-09 |0.5321750603676954       | 0.704540645696027         |
| VIF       |  0.999999   |0.02510546643737374      | 0.020872796619184383      |
    
 As you can see ,these matrics alone didn't work to find the similarity as any slight change in camera angle or lighting condition would give the result as the two images are not simillar.
 But the mertics SSIM and MSSSIM showed promise.
 
## 2 Using Principle component analysis.
Computed a 2 component PCA of all the images and checked if simillar images can be clustered. But as you see in the plot similar images (same coloured dots) are spread randomly.
This could be beacause all the images are somewhat similar due to the fact that all of them are pictures of cows in a simillar environment. 

![alt text](https://github.com/ashish-roopan/SIFT-image-simillarity/blob/main/pca.png)

Computing PCA with more components and clustering them based on some distance like cosine distance may work.
    
# 3. Using SIFT 
Due to the lack of more data I needed to go with a one shot learning method, which calculates a threshold value which classifies two images as similar or not similar.
Since the data consists of photos of same cow taken using similar camera angles and lighting condition, and the above shown  SSIM and MSSSIM were able to differentiate
similar and disimilar photos, I thought to use SIFT algorithm to match the structure in the both images.

### Steps
#### 1. Finding threshold
  - Detect keypoints and descriptors from a pair of 2 images.
  - Match these keypoints on both the images.
  - Find a similarity score defined by the below equation for this pair of images.
  ```
    distances = []
    for match in matches:
        sum += match.distance
        distances.append(match.distance)
    num_matches = len(matches)
    mean_distance = sum / num_matches
    score =  num_matches/max(len(keypoints_1), len(keypoints_2)) + 1/(mean_distance + 0.0001)
  ```
  I came up with this similarity scores because the first term ```num_matches/max(len(keypoints_1), len(keypoints_2))``` increases if the images are similar ,
  as the  ```num_matches``` will be more for similar images. And the second term ``` 1/(mean_distance + 0.0001) ``` will also increase for similar images as mean_distance approaches zero.
  - Do the above steps for all the similar images and find the average score (avg_similar_score).
  - And also for all the disimilar image pairs and find the average score (avg_dissimilar_score) .
  - Then the threshold can be found as ```thresh = (avg_similar_score + avg_dissimilar_score) / 2```
  
#### 2. Running inference
  - Detect keypoints and descriptors for a pair of image.
  - Match these keypoints on both the images.
  - Find a similarity score defined by the below equation for this pair of images.
  - If the similarity score is greater than the threshold the two images will be similar and vice versa.
  
#### Results using SIFT method
| False negatives |8    |
|-----------------|-----|
| False positives | 10  |
| True positives  | 104 |
| True negatives  | 1255|
| Precesion       | 0.91|
| Recall          | 0.93|
  
  
