import cv2
import tools

_img_Name = 'img/2.png'

_img = cv2.imread(_img_Name)

_h, _w, _d = _img.shape

_img_Grey = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
_thre, _bina = cv2.threshold(_img_Grey, 150, 255, cv2.THRESH_BINARY)
_canny = cv2.Canny(_img_Grey, 150, 255)

_cnts = cv2.findContours(_bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
_cnt = _cnts[0]
_biggestIndex = tools.findBiggestIndex(_cnt, 1)
print(_cnt[_biggestIndex].reshape(3027,2))
cv2.drawContours(_img, _cnt, _biggestIndex, [0,0,255])
# cv2.imshow('_bina',_bina)
# cv2.imshow('_canny',_canny)
cv2.imshow('img',_img)
test1
cv2.waitKey(0)
cv2.destroyAllWindows()