import cv2
import numpy as np
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

img1 = cv2.imread('data/23-09-20-cow1/IMG_20200923_134230_117.jpg')
img2 = cv2.imread('data/23-09-20-cow1/IMG_20200923_134237_170.jpg')
img3 = cv2.imread('data/23-09-20-cow22/IMG_20200923_135153_204.jpg')

img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))
img3 = cv2.resize(img3, (512, 512))

print("MSE: ", mse(img1,img1), mse(img1,img2), mse(img1,img3))
print("RMSE: ", rmse(img1, img1), rmse(img1, img2), rmse(img1, img3))
print("PSNR: ", psnr(img1, img1), psnr(img1, img2), psnr(img1, img3))
print("SSIM: ", ssim(img1, img1), ssim(img1, img2), ssim(img1, img3))
print("UQI: ", uqi(img1, img1), uqi(img1, img2), uqi(img1, img3))
print("MSSSIM: ", msssim(img1, img1), msssim(img1, img2), msssim(img1, img3))
print("ERGAS: ", ergas(img1, img1), ergas(img1, img2), ergas(img1, img3))
print("SCC: ", scc(img1, img1), scc(img1, img2), scc(img1, img3))
print("RASE: ", rase(img1, img1), rase(img1, img2), rase(img1, img3))
print("SAM: ", sam(img1, img1), sam(img1, img2), sam(img1, img3))
print("VIF: ", vifp(img1, img1), vifp(img1, img2), vifp(img1, img3))

img = np.hstack((img1, img2, img3))
img = cv2.resize(img, (1200, 400 ))
cv2.imshow('image', img)
cv2.waitKey(0)

cv2.imwrite('data/compare_matrics.jpg', img)