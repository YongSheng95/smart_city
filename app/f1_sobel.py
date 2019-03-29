# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import copy
def platelocate(img_path):
    resultRects = []
    img_crops = []
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    # img = cv2.imread(img_path)
    if debug < 3:
        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    contours = sobelLocate(img) + mserLocate(img) + colorLocate(img)
    # contours = mserLocate(img)
    rects = []
    for contour in contours:
        minRect = cv2.minAreaRect(contour)
        if verifysizes(minRect):
            rects.append(minRect)
    compareinfo = []
    for rect in rects:
        if verifysizes(rect):
            # 根据矩形转成box类型，并int化
            width = rect[1][0]
            height = rect[1][1]
            dsize = (int(width), int(height))
            ratio = float(width) / float(height)
            angle = rect[2]
            if ratio < 1:
                angle = angle + 90
                dsize = (int(height), int(width))
            if (angle < 30) and (angle > -30):
                rotmat = cv2.getRotationMatrix2D(rect[0], angle, 1)
                img_rotated = cv2.warpAffine(img, rotmat, (img.shape[1], img.shape[0]))
                resultRect, imgcrop = showresultrect(img_rotated, dsize, rect[0])
                resultRects.append(resultRect)
                img_crops.append(imgcrop)
                # cv2.imshow("test", resultRect)
                # cv2.waitKey(0)
                compareinfo.append((dsize, rect[0]))
    return resultRects, compareinfo, img_crops

def verifysizes(rect):
    area = int(rect[1][0]*rect[1][1])
    try:
        ratio = max(rect[1])/min(rect[1])
    except:
        return False
    return (area >= 3200) and (area <= 20000) and(ratio >= 1.3) and (ratio <= 5.0)

def showresultrect(img_rotated, dsize, center):
    img_crop = cv2.getRectSubPix(img_rotated, (dsize[0]+2, dsize[1]+3), center)
    img_test = cv2.resize(img_crop, (136, 36))
    return img_test, img_crop

def sobelLocate(img):
    # 高斯模糊：车牌识别中利用高斯模糊将图片平滑化，去除干扰的噪声对后续图像处理的影响
    gaussian = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    if debug == 1:
        cv2.imshow("img", gaussian)
        cv2.waitKey(0)
    # 灰度化
    gray = cv2.cvtColor(gaussian, cv2.COLOR_RGB2GRAY)
    if debug == 1:
        cv2.imshow("img", gray)
        cv2.waitKey(0)
    # equal = cv2.equalizeHist(gray)
    # if debug == 1:
    #     cv2.imshow("equal", equal)
    #     cv2.waitKey(0)
    # sobel算子：车牌定位的核心算法，水平方向上的边缘检测，检测出车牌区域
    sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))
    sobely1 = cv2.convertScaleAbs(sobely)
    sobel = cv2.addWeighted(sobelx1, 0.9, sobely1, 0.1, 0)
    if debug == 1:
        cv2.imshow("img", sobel)
        cv2.waitKey(0)
    # 进一步对图像进行处理，强化目标区域，弱化背景。
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    if debug == 1:
        cv2.imshow("img", binary)
        cv2.waitKey(0)
    # 进行开操作，去除细小噪点
    # eroded = cv2.erode(binary, None, iterations=1)
    # dilation = cv2.dilate(binary, None, iterations=1)
    # if debug == 1:
    #     cv2.imshow("dilation", dilation)
    #     cv2.waitKey(0)

    # 进行闭操作，闭操作可以将目标区域连成一个整体，便于后续轮廓的提取
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    if debug == 1 or debug == 2:
        cv2.imshow("img", closed)
        cv2.waitKey(0)

    # 寻找轮廓

    image, contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = copy.copy(img)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 0)
    if debug == 1 or debug == 2:
        cv2.imshow("img", result)
        cv2.waitKey(0)
    return contours

def colorLocate(img):
    # define the range of blue and  yellow value
    lower_blue = np.array([100, 110, 110])
    upper_blue = np.array([130, 255, 255])
    # lower_yellow = np.array([15, 55, 55])
    # upper_yellow = np.array([50, 255, 255])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    res = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("closed", closed)
    # cv2.waitKey(0)

    _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    result = copy.copy(img)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 0)
    # if debug == 1 or debug == 2:
    #     cv2.imshow("closed", result)
    #     cv2.waitKey(0)
    return contours

def mserLocate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # get mser, and set parameters
    mser1 = cv2.MSER_create(6, 3000, 14000, 0.5, 0.3)
    # detect plate regions
    # MSER+ detection
    coords1, _ = mser1.detectRegions(gray)
    # MSER- detection
    # coords2, _ = mser2.detectRegions(gray_neg)
    # mser_map = np.zeros(gray.shape)
    # mser_neg_map = np.zeros(gray.shape)
    contours = []
    for coord in coords1:
        x, y, w, h = cv2.boundingRect(coord)
        if w / h < 1.30 or w / h > 4.0:
            continue

        contours.append(coord)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.imshow("122", img)
    return contours

debug = 3

if __name__ == '__main__':
    pass














