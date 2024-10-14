import cv2 as cv
import numpy as np
from PIL import Image
from skimage import util

#gaussian：高斯噪声
img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='gaussian')

cv.imshow("source", img)
cv.imshow("lenna",noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()

#poisson：泊松噪声
img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='poisson')

cv.imshow("source", img)
cv.imshow("lenna",noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()

#pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='pepper')

cv.imshow("source", img)
cv.imshow("lenna",noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()
