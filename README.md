# SIFT-image-simillarity
Calculate similarity between 2 images using SIFT 

# Usage
## Testing two image for similarity
```
python check_similarity.py --thresh 0.00013 -i1 img1_path -i2 img2_path
````
The images will be similar if score > 0.00013





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

