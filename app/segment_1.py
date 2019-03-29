# -*- encoding: utf-8 -*-
import os
import cv2
import numpy as np
import math

# const param configure
def process(gray_img, size=(136, 36)):
    img_resize = cv2.resize(gray_img, size)
    gray_gauss = cv2.GaussianBlur(img_resize, (3, 3), 0)
    ret, thresh = cv2.threshold(gray_gauss, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count = np.sum(thresh)/255
    size = 136*36
    if count/size >= 0.4:
        thresh = ~thresh
    count_change = []
    for i in range(0, 36):
        count_change.append(0)
    for raw in range(0, 36):
        list_ = thresh[raw, :]
        now = list_[0]
        change = False
        for i, data in enumerate(list_):
            if now == data:
                change = False
            if now != data:
                change = True
                count_change[raw] = count_change[raw] + 1
            now = data

    # print(count_change)
    for i, count in enumerate(count_change):
        if i < 2 or i > 34:
            thresh[i, :] = 0
        elif i < 4 or i > 31:
            if i < 35:
                if count < 25 and count_change[i+1] < 10:
                    thresh[i, :] = 0

                if count < 21 and count_change[i+1] < 10:
                    thresh[i, :] = 0
                if count < 15 and count_change[i+1] < 10:
                    thresh[i, :] = 0
            if count < 14:
                thresh[i, :] = 0
        elif i < 7 or i > 29:
            if i < 35:
                if count < 16 and count_change[i + 1] < 10:
                    thresh[i, :] = 0
            if count < 13:
                thresh[i, :] = 0
        else:
            if count < 12 and i < 35 and count_change[i+1] < 10:
                thresh[i, :] = 0
            if count < 10:
                thresh[i, :] = 0
    return thresh

#去小块噪声
def delete_dot(thresh):
    count_pixel = np.sum(thresh, axis=0) / 255
    find_dot = []
    last = 0
    temp = 0.
    for i, data in enumerate(count_pixel):
        if data == 0 and last == 0:
            last = data
            continue
        if data != 0 and last == 0:
            temp = temp + data
            last = data
            continue
        if data == 0 and last != 0:
            find_dot.append((temp, i-1))
            last = data
            temp = 0
            continue
        if data != 0 and last != 0:
            temp = temp + data
            last = data
            continue
    if count_pixel[-1] != 0:
        find_dot.append((temp, 135))

    ready_cut = []
    delete_indexes = []
    for x in find_dot:
        if x[0] < 42 and (x[1] < 5 or x[1] > 131):
            delete_indexes.append(x)
        elif x[0] < 66 and (45 < x[1] < 56):
            delete_indexes.append(x)
        elif x[0] < 30:
            delete_indexes.append(x)
        else:
            ready_cut.append(x)

    result1 = sorted(find_dot, key=lambda x: x[1])
    # print(delete_indexes)
    for result in delete_indexes:
        index = result[1]
        for i in range(index, -1, -1):
            if count_pixel[i] != 0:
                thresh[:, i] = 0
            if count_pixel[i] == 0:
                break
    # print(count_pixel)
    # print(result1)
    # print(ready_cut)
    return thresh, ready_cut

def gety1_y2(zone):
    y2 = 36
    y1 = 0
    # list = result[:, zone[0]: zone[1]]
    # for i, data in enumerate(list.tolist()):
    #     if data > 0:
    #         if i <= y1:
    #             y1 = i
    #         if i <= y2:
    #             y2 = i
    return 2, 33
def getstart(ready_cut):
    index = -1
    for dot in ready_cut:
        if 35 < dot[1] < 50:
            index = ready_cut.index(dot)
            break
        else:
            continue
    if index == -1:
        return 0, 0
    else:
        return index, index + 1

def find_left_letter(left_list):
    end = -1
    cut = False
    count = 0
    left_letters_in = []
    for i, char in enumerate(reversed(left_list)):

        if count < 3:
            width_now = char[1] - char[0] + 1
            if 9 <= width_now <= 18:
                left_letters_in.append(char)
                count = count + 1
                continue
            if width_now > 18:
                left_letters_in.append((char[0] + 1 + int(width_now / 5*3), char[1]))
                left_letters_in.append((char[0], char[0] + int(width_now / 5 * 3)))
                count = count + 2
        else:
            break
    return list(reversed(left_letters_in))

def find_right_letter(right_list, platetype=7):
    count = 0
    right_letters_in = []
    if len(right_list) == 5:
        return right_list
    else:
        for i, char in enumerate(right_list):
            width_now = char[1] - char[0] + 1
            if count < 5:
                if width_now <= 20:
                    right_letters_in.append(char)
                    count = count + 1
                    continue
                if width_now < 9:
                    pass
                if width_now > 20:
                    pass
            else:
                break
        return right_letters_in

def first_cut(ready_cut, img):
    vertical_pixel = np.sum(img, axis=0) / 255
    # print(vertical_pixel)
    first_cut_list = []
    for i, dot in(enumerate(ready_cut)):
        end = dot[1]
        start = end
        for m in reversed(range(0, end+1)):
            if vertical_pixel[m] != 0 and m > 0:
                continue
            if vertical_pixel[m] == 0:
                start = m + 1
                first_cut_list.append((start, end))
                break
            if m == 0 and vertical_pixel[m] != 0:
                start = m
                first_cut_list.append((start, end))
                break
    return first_cut_list

def cut(cut_list, result):
    seg_list = []
    # print(cut_list[1])
    cut_list[1] = (cut_list[1][0]+5, cut_list[1][1]-2)
    # print(cut_list[1])
    for zone in cut_list:
        y1, y2 = gety1_y2(zone)
        if (zone[1] - zone[0] < 6) and (zone[1] + 4 < 136) and(zone[0] > 4):
            seg_list.append(cv2.resize(result[y1:y2, zone[0] - 4:zone[1] + 4],
                                       (16, 28)))
        else:
            if (zone[0] > 2) and (zone[1] < 134):
                seg_list.append(cv2.resize(result[y1:y2, zone[0]-2:zone[1]+2],
                                           (16, 28)))
            else:
                seg_list.append(cv2.resize(result[y1:y2, zone[0]:zone[1]],
                                           (16, 28)))
    return seg_list

def cut_run_wj(gray_img, platetype=7):
    # 去上下边框
    thresh = process(gray_img, (136, 36))
    thresh_copy = thresh.copy()
    # 去小块噪声
    result, ready_cut = delete_dot(thresh_copy)
    # 确定字符左右边界
    first_cut_list = first_cut(ready_cut, result)

    if len(first_cut_list) == 7:
        seg_list = cut(first_cut_list, result)
        start_left = 1
        start_right = 2
        left_letters = find_left_letter(first_cut_list[0:start_left + 1])
        right_letters = find_right_letter(first_cut_list[start_right:], 7)
    else:
        start_left, start_right = getstart(ready_cut)
        left_letters = find_left_letter(first_cut_list[0:start_left + 1])
        right_letters = find_right_letter(first_cut_list[start_right:], platetype)
        seg_list = cut(left_letters + right_letters, result)
    # print(platetype, len(seg_list))
    if len(seg_list) == 0 or len(seg_list) != 8:
        return ['UnSegment']
    else:
        temp = seg_list[2]
        seg_list[2] = seg_list[1]
        seg_list[1] = temp

        temp = seg_list[1]
        seg_list[1] = seg_list[0]
        seg_list[0] = temp
        return seg_list



