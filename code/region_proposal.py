import numpy as np
import cv2
 
cv2.setUseOptimized(True);
cv2.setNumThreads(8);


img = cv2.imread('combined/IMG_20200922_160407_838.jpg')
img = cv2.resize(img, (512, 512))
H, W, C = img.shape

# instanciate the selective search
# segmentation algorithm of opencv
search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
# set the base image as the input image
search.setBaseImage(img)
 
# since we'll use the fast method we set it as such
 
# you can also use this for more accuracy:
search.switchToSelectiveSearchFast()
rects = search.process()  # process the image
print('Total Number of Region Proposals: {}'.format(len(rects)))

roi = img.copy()
for (x, y, w, h) in rects:
    
    # Check if the width and height of
    # the ROI is atleast 10 percent
    # of the image dimensions and only then
    # show it
    if (w / float(W) < 0.8 or h / float(H) < 0.8):
        continue
 
    # Let's visualize all these ROIs
    color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
    cv2.rectangle(roi, (x, y), (x + w, y + h),
                  color, 2)
 
roi = cv2.resize(roi, (640, 640))
final = cv2.hconcat([cv2.resize(img, (640, 640)), roi])
cv2.imshow('ROI', final)
cv2.waitKey(0)