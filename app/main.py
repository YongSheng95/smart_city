import os
import csv
import cv2
import numpy as np
import xlsxwriter
from app.location import licensePlateLocate
from app.segment import char_segment
from app.recognition import recognise


def run():
    path = 'img_test'
    files = load_image_file(path)
    results = []
    name_list = []
    seg_img_list = []
    color_img_list = []
    rec_count = 0
    BATCH = 100

    for file in files:
        # file_name = file.split('\\')[-1].split('.')[0]
        # locate
        file_type = jude_image_type(file)
        color_img, img = licensePlateLocate(file, file_type)

        # locate fail
        if img is None:
            color_img = ['UnLocated']
            char_seg_img = ['UnLocated']
        else:
            print(file)
            try:
                # segment
                char_seg_img =  char_segment(img)
            except:
                char_seg_img = ['UnSegment']
            # save segment
            # if len(char_seg_img) != 1:
            #     save_segment_and_color_image(char_seg_img, color_img, file_name)
            # else:
            #     print(char_seg_img)

            # if len(char_seg_img) != 1:
            #     for a_img in char_seg_img:
            #         cv2.imshow('color', color_img)
            #         cv2.imshow('a_img', a_img)
            #         cv2.waitKey(0)
            # else:
            #     print(char_seg_img)

        # add list
        name_list.append(file)
        seg_img_list.append(char_seg_img)
        color_img_list.append(color_img)
        rec_count += 1
        # recognise
        if len(seg_img_list) == BATCH:
            print('count:',rec_count)
            result = recognise(color_img_list, seg_img_list)
            # print(result)
            results.extend(result)
            # clear batch list
            seg_img_list = []
            color_img_list = []

    # last batch recognise
    if len(seg_img_list) != 0:
        result = recognise(color_img_list, seg_img_list)
        results.extend(result)

    # save to excel
    print('rec_count: ',rec_count)
    print('len results: ',len(results))
    recognise_results = zip(results, name_list)
    write_to_excel(recognise_results)




def load_image_file(path):
    for dir_path, dir_names, file_names in os.walk(path):
        for file in file_names:
            if file.endswith('.jpg' or '.jpeg'):
                file_path = os.path.join(dir_path, file)
                yield file_path


def write_to_excel(files):
    file_path = '../result/test_6.xlsx'
    work_book = xlsxwriter.Workbook(file_path)
    data_sheet = work_book.add_worksheet('sheet1')
    row_name = [u'车牌号', u'车牌颜色', u'测试文件名']
    for i in range(len(row_name)):
        data_sheet.write(0, i, row_name[i])
    for i, item in enumerate(files):
        data_sheet.write(i + 1, 0, item[0][0])
        data_sheet.write(i + 1, 1, item[0][1])
        data_sheet.write(i + 1, 2, item[1])
    work_book.close()


def save_segment_and_color_image(seg_chars_img, color_img, name):
    step = 1
    for char_img in seg_chars_img:
        cv2.imencode('.jpg', char_img)[1].tofile(
            '../result/temp/seg_chars/{0}_of_{1}.jpg'.format(name, step))
        step += 1
    cv2.imencode('.jpg', color_img)[1].tofile(
        '../result/temp/color/{0}.jpg'.format(name))

def jude_image_type(file_path):
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    if len(contours) < 100:
        file_type = 1
    else:
        file_type = 2
    return file_type



if __name__ == '__main__':
    run()


