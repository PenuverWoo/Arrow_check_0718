import cv2
import tools
import matplotlib.pyplot as plt
import math
import numpy as np

_img_Name = 'img/5.png'
_img1 = cv2.imread(_img_Name, 0)
_img = cv2.imread(_img_Name)

# dst = cv2.pyrMeanShiftFiltering(_img1, 10, 50)
# Laplacian =  cv2.Laplacian(_img1, cv2.CV_16S, ksize=3)
# dst = cv2.convertScaleAbs(Laplacian)

_h, _w, _d = _img.shape

_img_Grey = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
_img_Grey_Copy = _img_Grey.copy()
_img_Grey_Copy = cv2.cvtColor(_img_Grey_Copy, cv2.COLOR_GRAY2BGR)

kernel_sharpen = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1],
        [-1,-1,-1,-1,-1]])/8.0
# kernel_sharpen = np.array([
#         [-1,-1,-1],
#         [-1,9,-1],
#         [-1,-1,-1]])
# kernel_sharpen = np.array([
#         [1,1,1],
#         [1,-8,1],
#         [1,1,1]])
_img_Grey = cv2.filter2D(_img_Grey,-1,kernel_sharpen)
_thre, _bina = cv2.threshold(_img_Grey, 100, 255, cv2.THRESH_BINARY)
# _thre, _bina = cv2.threshold(_img_Grey[int(_h/2-100):int(_h/2+100), int(_w/2-100):int(_w/2+100)], 110, 255, cv2.THRESH_BINARY)
# _canny = cv2.Canny(_img_Grey, 100, 255)

_cnts = cv2.findContours(_bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
cv2.drawContours(_img_Grey_Copy, _cnts, -1, [0,0,255])

# the biggest area coords

_MaxArea = max(_cnts, key=cv2.contourArea)
((x, y), _radius) = cv2.minEnclosingCircle(_MaxArea)
_center = (math.ceil(x)-4, math.ceil(y)-3)

# ROI in Arrow effect

ROI_CenterOfArrow = tools.ROI(_img, _center, _radius, 1)
_ROI_cnts = cv2.findContours(ROI_CenterOfArrow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
_q1 = cv2.cvtColor(ROI_CenterOfArrow, cv2.COLOR_GRAY2RGB)

# Draw ROI area

# cv2.drawContours(_img, _ROI_cnts, 1, [0,0,255])
# cv2.circle(_img, _center, 41, (0,0,255))

oct_Poly, oct_Coords = tools.drawOctagon(_img_Grey_Copy, _center, _radius, 0.18, 22.5)   # Draw Octagon into image
cv2.imshow('oct_Poly',oct_Poly)
print(oct_Coords)
# the biggest area index


# _biggestIndex = tools.findBiggestIndex(_cnts, 2)
# print(_cnt[_biggestIndex].reshape(3027,2))
# cv2.drawContours(_img, _cnts, 18, [0,0,255])

# Check the pionts whether on the Octgaton
dist = cv2.pointPolygonTest(_ROI_cnts[1], (213,273), True)
op = tools.checkInScale(oct_Coords, _ROI_cnts[1])
print(op)

cv2.imshow('_img_Grey_Copy',_img_Grey_Copy)

cv2.imshow('_img_Grey',_img_Grey)

_img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
plt.imshow(_img_Grey_Copy)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()



# Nest step, calcul how much pixel mathed between centre-edge and Octgaton, and then shfit and rotate to figure out the best position for Octgaton