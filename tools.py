import cv2

def findBiggestIndex(cnt, num):
    biggestArea = 0
    biggestIndex = 0
    secondArea = 0
    secondIndex = 0
    for i in range(len(cnt)):
        area = cv2.contourArea(cnt[i])
        if area > biggestArea:
            secondArea = biggestArea
            secondIndex = biggestIndex
            biggestArea = area
            biggestIndex = i
    if num == 1:
        return biggestIndex
    if num == 2:
        return secondIndex
