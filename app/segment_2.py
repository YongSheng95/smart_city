import cv2
import os
import numpy as np

def process(gray_img, size=(136, 36)):
    img_resize = cv2.resize(gray_img, size)
    gray_gauss = cv2.GaussianBlur(img_resize, (3, 3), 0)
    ret, thresh = cv2.threshold(gray_gauss, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count = np.sum(thresh)/255
    size = 100*60
    if count/size >= 0.4:
        thresh = ~thresh

    count_change = []
    for i in range(0, 60):
        count_change.append(0)
    for raw in range(0, 60):
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
        if i < 2 or i > 57:
            thresh[i, :] = 0
        elif i < 4:
            # if i < 35:
            #     if count < 25 and count_change[i+1] < 10:
            #         thresh[i, :] = 0
            #     if count < 21 and count_change[i+1] < 10:
            #         thresh[i, :] = 0
            #     if count < 15 and count_change[i+1] < 10:
            #         thresh[i, :] = 0
            if count < 16:
                thresh[i, :] = 0
        elif i > 55:
            if count < 21:
                thresh[i, :] = 0
        # elif i < 7 or i > 29:
        #     if i < 35:
        #         if count < 16 and count_change[i + 1] < 10:
        #             thresh[i, :] = 0
        #     if count < 13:
        #         thresh[i, :] = 0
        # else:
        #     # if count < 12 and i < 35 and count_change[i+1] < 10:
        #     #     thresh[i, :] = 0
        #     if count < 6:
        #         print(i)
        #         thresh[i, :] = 0
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
        find_dot.append((temp, 99))

    ready_cut = []
    delete_indexes = []
    for x in find_dot:
        if x[1] < 15 or x[1] > 90 and x[0] < 121:
            delete_indexes.append(x)
        elif x[0] < 50:
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

def find_up_letters(cut_list):
    if len(cut_list) == 2:
        return cut_list
    if len(cut_list) < 2:
        return [(27, 45), (54, 70)]
    else:
        for i, char in enumerate(cut_list):
            if char[0] > 49:
                index = i - 1
                if index != 0 and (cut_list[index][1]-cut_list[index][0]) < 9:
                    return [(cut_list[index-1][0], cut_list[index][1]), cut_list[index+1]]
                else:
                    return [cut_list[index], cut_list[index + 1]]

    return []

def find_down_letters(cut_list):
    count = 0
    left_letters = []
    for i, char in enumerate(cut_list):
        if count < 5:
            width_now = char[1] - char[0] + 1
            if 9 <= width_now <= 20:
                left_letters.append(char)
                count = count + 1
                continue
            if width_now > 20:
                if char[0] == 0:
                    left_letters.append((char[0] + 1 + int(width_now / 2), char[1]))
                    count = count + 1
                else:
                    left_letters.append((char[0], char[0] + int(width_now / 2)))
                    count = count + 2
        else:
            break
    return left_letters

def cut(cut_list, result):
    seg_list = []
    if len(cut_list) == 2:
        y1, y2 = 1, 20
    elif len(cut_list) == 5:
        y1, y2 = 1, 36
    else:
        return []
    for zone in cut_list:
        if (zone[1] - zone[0] < 6) and (zone[1] + 4 < 100) and(zone[0] > 4):
            seg_list.append(cv2.resize(result[y1:y2, zone[0] - 4:zone[1] + 4],
                                       (16, 28)))
        else:
            if (zone[0] > 2) and (zone[1] < 98):
                seg_list.append(cv2.resize(result[y1:y2, zone[0]-2:zone[1]+2],
                                           (16, 28)))
            else:
                seg_list.append(cv2.resize(result[y1:y2, zone[0]:zone[1]],
                                           (16, 28)))
    return seg_list

def cut_run_double(img_gray):
    thresh = process(img_gray, (100, 60))
    layer1 = thresh[0:22, :]
    layer2 = thresh[22:60, :]
    # thresh_copy = thresh.copy()
    # # 去小块噪声
    result1, ready_cut1 = delete_dot(layer1)
    result2, ready_cut2 = delete_dot(layer2)
    first_cut_list1 = first_cut(ready_cut1, result1)
    first_cut_list2 = first_cut(ready_cut2, result2)
    up_letters = find_up_letters(first_cut_list1)
    down_letters = find_down_letters(first_cut_list2)
    if len(up_letters) == 2 and len(down_letters) == 5:
        seg_list_up = cut(up_letters, result1)
        seg_list_down = cut(down_letters, result2)
        # cv2.imshow("gr", thresh)
        # cv2.waitKey(0)
        if len(seg_list_up) == 2 and len(seg_list_down) == 5:
            return seg_list_up + seg_list_down
    return ['UnSegment']

if __name__ == "__main__":
    # 中文路径读取会出错
    path = "./segment_bigback"
    for dir_path, dir_names, file_names in os.walk(path):
        for file in file_names:
            # print(file)
            file_path = os.path.join(dir_path, file)
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cut_run_double(img_gray)

