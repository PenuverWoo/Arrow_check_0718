import cv2
import tools
import matplotlib.pyplot as plt

_img_Name = 'img/2.png'

_img = cv2.imread(_img_Name)

_h, _w, _d = _img.shape

_img_Grey = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
_thre, _bina = cv2.threshold(_img_Grey, 140, 255, cv2.THRESH_BINARY)
# _thre, _bina = cv2.threshold(_img_Grey[int(_h/2-100):int(_h/2+100), int(_w/2-100):int(_w/2+100)], 110, 255, cv2.THRESH_BINARY)
# _canny = cv2.Canny(_img_Grey, 100, 255)

_cnts = cv2.findContours(_bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]

_MaxArea = max(_cnts, key=cv2.contourArea)      # the biggest area coords

((x, y), _radius) = cv2.minEnclosingCircle(_MaxArea)
_center = (int(x)-1, int(y)+2)

cv2.circle(_img, _center, 43, (0,0,255))

_biggestIndex = tools.findBiggestIndex(_cnts, 2) # the biggest area index
# print(_cnt[_biggestIndex].reshape(3027,2))
# cv2.drawContours(_img, _cnts, 18, [0,0,255])

cv2.imshow('_img',_img)
# cv2.imshow('_canny',_canny)
cv2.imshow('_bina',_bina)

_img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
plt.imshow(_img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()