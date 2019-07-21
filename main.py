import cv2
import tools
import matplotlib.pyplot as plt
import math
import numpy as np

_img_Name = 'img/5.png'
_img1 = cv2.imread(_img_Name)
_img = cv2.imread(_img_Name)

_h, _w, _d = _img.shape

_img_Grey = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
_thre, _bina = cv2.threshold(_img_Grey, 100, 255, cv2.THRESH_BINARY)
# _thre, _bina = cv2.threshold(_img_Grey[int(_h/2-100):int(_h/2+100), int(_w/2-100):int(_w/2+100)], 110, 255, cv2.THRESH_BINARY)
# _canny = cv2.Canny(_img_Grey, 100, 255)

_cnts = cv2.findContours(_bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
cv2.drawContours(_img, _cnts, -1, [0,0,255])

# the biggest area coords

_MaxArea = max(_cnts, key=cv2.contourArea)
((x, y), _radius) = cv2.minEnclosingCircle(_MaxArea)
_center = (math.ceil(x)-3, math.ceil(y))

# ROI in Arrow effect

ROI_CenterOfArrow = tools.ROI(_img, _center, _radius, 1)
_ROI_cnts = cv2.findContours(ROI_CenterOfArrow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
_q1 = cv2.cvtColor(ROI_CenterOfArrow, cv2.COLOR_GRAY2RGB)

# Draw ROI area

cv2.drawContours(_img, _ROI_cnts, 1, [0,0,255])
# cv2.circle(_img, _center, 41, (0,0,255))

oct_Poly = tools.drawOctagon(_img, _center, _radius, 0.18, 22.5)   # Draw Octagon into image
cv2.imshow('oct_Poly',oct_Poly)

# the biggest area index

# _biggestIndex = tools.findBiggestIndex(_cnts, 2)
# print(_cnt[_biggestIndex].reshape(3027,2))
cv2.drawContours(_img, _cnts, 18, [0,0,255])


cv2.imshow('_img1',_img1)

_img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
plt.imshow(_img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

# Nest step, calcul how much pixel mathed between centre-edge and Octgaton, and then shfit and rotate to figure out the best position for Octgaton