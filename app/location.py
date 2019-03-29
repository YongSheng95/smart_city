#-*- encoding:utf-8 -*-
import numpy as np
import cv2
from libsvm.python.svmutil import *
from libsvm.python.svm import *
from app.f1_sobel import platelocate
from app.f1_sobel_1 import platelocate_1
from app.j_square import j_s

model = svm_load_model("../src/7-3-libsvm")

def licensePlateLocate(img_path, file_type):
    if file_type == 2:
        result_rects, compare_info, img_crops = platelocate(img_path)
        result_prob = []
        if len(result_rects) != 0:
            for result in result_rects:
                result_prob.append(judge_prob(result))
            if len(result_prob) != 0:
                result_index = result_prob.index(max(result_prob))
                img_color = cv2.resize(img_crops[result_index], (100, 32))
                return img_color, img_crops[result_index]
        else:
            return None, None

    if file_type == 1:
        result_rects, compare_info, img_crops = platelocate_1(img_path)
        result_prob = []
        if len(result_rects) != 0:
            for result in result_rects:
                result_prob.append(judge_prob(result))
            if len(result_prob) != 0:
                result_index = result_prob.index(max(result_prob))
                img = img_crops[result_index]
                img_type = j_s(img)
                try:
                    if img_type == 'horizontal':
                        img = fix_horizontal(img)
                    if img_type == 'vertical':
                        img = fix_vertical(img)
                    if img_type == 'wrongcut' or img_type == 'god know':
                        img = fix_rd_light_level(img)
                        img = fix_affine(img)
                    img = fix_avg_light_level(img)
                except:
                    print('fix error.')

                return cv2.resize(img, (100, 32)), img
        else:
            return None, None


def judge_prob(result):
    data = getfeatures_prob(result)
    y_prob = predict_prob(data)
    return y_prob

def getrow(img_in, j):
    return img_in[j]
def getcol(img_in, j):
    col = []
    for i in range(img_in.shape[0]):
        col.append(img_in[i][j])
    return col
def projectedhistogram(img_in, string):
    sz = 0
    if string == "Horizontal":
        sz = img_in.shape[0]

    else:
        sz = img_in.shape[1]
    nonezeorimg = []
    img_in = cv2.extractChannel(img_in, 0)
    for j in range(sz):
        data = getrow(img_in, j) if (string == "Horizontal") else getcol(img_in, j)
        count = cv2.countNonZero(np.array(data))
        nonezeorimg.append(count)
    maxnum = 0.0
    for j in range(len(nonezeorimg)):
        maxnum = max(maxnum, nonezeorimg[j])
    if maxnum > 0:
        for j in range(len(nonezeorimg)):
            nonezeorimg[j] = nonezeorimg[j] / float(maxnum)
    return nonezeorimg

def getfeatures_prob(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img_in = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY, )
    vhist = projectedhistogram(img_in, "Vertical")
    hhist = projectedhistogram(img_in, "Horizontal")
    numcols = len(vhist) + len(hhist)
    out = []
    j = 0
    for i in range(len(vhist)):
        out.append(vhist[i])
    for i in range(len(hhist)):
        out.append(hhist[i])
    return out

def getfeatures_prob(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img_in = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY, )
    vhist = projectedhistogram(img_in, "Vertical")
    hhist = projectedhistogram(img_in, "Horizontal")
    numcols = len(vhist) + len(hhist)
    out = []
    j = 0
    for i in range(len(vhist)):
        out.append(vhist[i])
    for i in range(len(hhist)):
        out.append(hhist[i])
    return out

def predict_prob(sample):
    samples = []
    samples.append(sample)
    p_label, p_acc, p_val = svm_predict([1], samples, model, '-b 1 -q')
    return p_val[0][1]


def fix_affine(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rows, cols = img_gray.shape
    min_bottom = 2
    index = 0
    for i in range(min_bottom, 10):
        col_list = thresh[-i, :]
        if np.count_nonzero(~col_list) < 150:
            continue
        for item in col_list:
            if 255 - item == 0:
                index += 1
                continue
            else:
                break
        if index > 70:
            index = 0
            continue
        if cols - index < min_bottom:
            index = 0
            continue
        else:
            row_cnt = i
            break

    pst1_1 = [2, 2]
    pst1_2 = [cols - index, 2]
    pst1_3 = [index, rows - row_cnt]

    pst2_1 = [2, 2]
    pst2_2 = [cols-2, 2]
    pst2_3 = [2, rows-2]

    pts1 = np.float32([pst1_1, pst1_2, pst1_3])
    pts2 = np.float32([pst2_1, pst2_2, pst2_3])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def fix_horizontal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # cv2.imshow("img_color", img)
    # cv2.waitKey(0)
    y_start_1 = -1
    x_start = -1
    y_start_2 = -1
    for i in range(0, 5):
        list_pixel = binary[:, i].tolist()
        if np.count_nonzero(~binary[:, i]) < 20:
            continue
        for index, p in enumerate(list_pixel):
            if p == 0:
                y_start_1 = index
                x_start = i
                break
        if y_start_1 != -1:
            break
    height, width = img.shape[0], img.shape[1]
    # 原图中卡片的四个角点
    pts1 = np.float32([[x_start, y_start_1], [width, 0], [x_start, height-y_start_1], [width, height]])
    # 变换后分别在左上、右上、左下、右下四个点
    pts2 = np.float32([[0, 0], [136, 0], [0, 36], [136, 36]])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst1 = cv2.warpPerspective(img, M, (136, 36))
    return dst1


def fix_vertical(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # cv2.imshow("img_color", img)
    # cv2.waitKey(0)
    height, width = img.shape[0], img.shape[1]
    y_start_1 = -1
    x_start = -1
    for i in range(0, 5):
        list_pixel = binary[height - 1 - i, :].tolist()
        if np.count_nonzero(~binary[height - 1 - i, :]) < 100:
            continue
        for index, p in enumerate(list_pixel):
            if p == 0:
                y_start_1 = height -1 - i
                x_start = index
                break
        if y_start_1 != -1:
            break

    # 原图中卡片的四个角点
    pts1 = np.float32([[0, 0], [width, 0], [x_start, height], [width - x_start, height]])
    # 变换后分别在左上、右上、左下、右下四个点
    pts2 = np.float32([[0, 0], [136, 0], [0, 36], [136, 36]])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst1 = cv2.warpPerspective(img, M, (136, 36))
    return dst1

def fix_avg_light_level(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    avg = np.average(gray)
    cv2.convertScaleAbs(img, img, 110 / avg)
    return img

def fix_rd_light_level(img):
    rows, cols = img.shape[0], img.shape[1]
    if rows/2 - int(rows/2) > 0:
        row = int(rows/2) + 1
    else:
        row = int(rows/2)
    img_up = img[0:row-1, :]
    img_down = img[row-1:, :]

    gray_up = cv2.cvtColor(img_up, cv2.COLOR_RGB2GRAY)
    avg_up = np.average(gray_up)
    gray_down = cv2.cvtColor(img_down, cv2.COLOR_RGB2GRAY)
    avg_down = np.average(gray_down)

    cv2.convertScaleAbs(img_up, img_up, avg_down / avg_up)
    return img


if __name__=="__main__":
    # 中文路径读取会出错
    import os
    # path = "..\src\Task3_车牌识别\性能评测图像库\竖直错切角变化子库\错切50"
    path = "img_test\性能评测图像库\典型竖直透视角变化子库"
    for dir_path, dir_names, file_names in os.walk(path):
        for file in file_names:
            file_path = os.path.join(dir_path, file)
            img_color, img = licensePlateLocate(file_path, 1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            print(img_gray.shape)
            # dst = fix_affine(img)
            dst = fix_vertical(img)
            cv2.imshow('img', img)
            cv2.imshow('dst', dst)
            cv2.waitKey(0)
