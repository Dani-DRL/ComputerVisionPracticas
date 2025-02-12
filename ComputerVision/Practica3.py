import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread("/Users/dani/Downloads/imagenes/ivvi_512x512_gray.jpg")
imSoCa = cv2.imread("/Users/dani/Downloads/imagenes/ivvi_684x684_gray.jpg")

if imagen is None:
    print("La imagen no se ha cargado correctamente")
    exit()


blur = cv2.blur(imagen, (5, 5))
gaussian = cv2.GaussianBlur(imagen, (5, 5), 0)
bilateral = cv2.bilateralFilter(imagen, 9, 75, 75)
median = cv2.medianBlur(imagen, 5, 0)

cv2.imshow('Imagen original', imagen)
cv2.imshow('Blur', blur)
cv2.imshow('Gaussian', gaussian)
cv2.imshow('Bilateral', bilateral)
cv2.imshow('Median', median)

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)

f2d = cv2.filter2D(imagen, -1, kernel)
cv2.imshow('Filtro 2d', f2d)

sobel_x = cv2.Sobel(imSoCa, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
sobelAbs_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.Sobel(imSoCa, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
sobelAbs_y = cv2.convertScaleAbs(sobel_y)
sobelAbs = cv2.addWeighted(sobelAbs_x, 0.5, sobelAbs_y, 0.5, 0)
_, sobelThreshold = cv2.threshold(sobelAbs, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('Sobel', sobelAbs)
cv2.imshow('Sobel Threshold', sobelThreshold)

canny = cv2.Canny(imSoCa, threshold1=100, threshold2=200, apertureSize=3, L2gradient=False)
cv2.imshow('Canny', canny)

cv2.waitKey()
cv2.destroyAllWindows()



exit(0)
