import cv2
import numpy as np
import math

def findBiggestIndex(cnt, num):                                 # Find out biggest or secend area's index or area in cnts
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

def ROI(img, center, radius, num):                              # 1 means ROI of center of Arrows
    temp = np.ones(img.shape[:2], np.uint8) * 0
    if num is 1:
        # print(type(center[0]))
        temp[center[1]-50:center[1]+50, center[0]-50:center[0]+50] = 255

        masked = cv2.bitwise_and(img, img, mask=temp)

        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        _ROI_thre, _ROI_bina = cv2.threshold(masked, 50, 255, cv2.THRESH_BINARY)
        return _ROI_bina

def drawOctagon(img, center, radius, RIO_ratio, rotate_Angle):  #Draw octagon in image image with diff angle
    x = center[0]
    y = center[1]
    InitAngle = 0
    locationX, LocationY = [ ((x - radius*RIO_ratio - x) * np.cos(np.radians(rotate_Angle)) + (y - y) * np.sin(np.radians(rotate_Angle)) + x), \
                            ((y - y) * np.cos(np.radians(rotate_Angle)) - (x - radius*RIO_ratio - x) * np.sin(np.radians(rotate_Angle)) + y)]

    octagon_piont = [[int(x), int(y)], [int(x), int(y)], [int(x), int(y)],
                     [int(x), int(y)],
                     [int(x), int(y)], [int(x), int(y)], [int(x), int(y)],
                     [int(x), int(y)]]
                                                                # old version for locate eight points, but now use hexagon_piont way, use it to hexagon_point for  array frame.

    for i in range(8):                                          # count eight points of hexagon after the init start_x,start_y changed by Mangle as LocationX/LocationY
        octagon_piont[i] = [math.ceil((locationX - x) * np.cos(np.radians(InitAngle)) + (LocationY - y ) * np.sin(np.radians(InitAngle)) + x), \
                            math.ceil((LocationY - y) * np.cos(np.radians(InitAngle)) - (locationX - x) * np.sin(np.radians(InitAngle)) + y)] # noticed the round way,the value would be lost 0 to 1
        InitAngle += 45
    octagon = np.array([[octagon_piont[0], octagon_piont[1], octagon_piont[2], octagon_piont[3], octagon_piont[4],
                         octagon_piont[5], octagon_piont[6], octagon_piont[7]]], dtype=np.int32)

    # Catch all of octagon coords
    temp = np.ones(img.shape[:2], np.uint8) * 0
    cv2.polylines(temp, octagon, 1, (255, 255, 255))
    cnts = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    # cv2.imshow('tep',temp)

    temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)
    cv2.polylines(img, octagon, 1, (0, 255, 0))
    return img, octagon[0], cnts[0]

def checkInScale(coord, cnts):                                 # check the points whether inside of the contours
    for i in range(len(coord)):
        value = cv2.pointPolygonTest(cnts, (coord[i][0], coord[i][1]), True)
        if value < 0:
            return [coord[i][0], coord[i][1]]
    return value

def CountInAreas(cnts, coords):
    # cnts1 = cnts1.reshape(len(cnts1), 2)
    X_num = 0
    Y_num = 0
    coords = coords.T
    coords[0] -= X_num
    coords = coords.T
    coords = coords.reshape(len(coords), 2)
    print(coords)
    # print(cnts1, cnts2)
    count = 0
    for i in range(len(coords)):
        value = cv2.pointPolygonTest(cnts, (coords[i][0], coords[i][1]), True)
        if value < 0:
            count += 1
    print(count)