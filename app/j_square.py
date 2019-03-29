import cv2
import numpy as np

def getflag(sq):
    rate = np.count_nonzero(sq)/100
    if rate > 0.4:
        #为白色
        return 1
    else:
        #为黑色
        return 0

def j_s(img):
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except:
        return 'god know'
    rows, cols = img_gray.shape[0], img_gray.shape[1]
    gray_gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)
    ret, thresh = cv2.threshold(gray_gauss, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sq_left_up = thresh[0:10, 0:10]
    sq_left_down = thresh[rows-10:, 0:10]
    sq_right_up = thresh[0: 10, cols-10:]
    sq_right_down = thresh[rows-10:, cols-10:]

    flag_lu = getflag(sq_left_up)
    flag_ld = getflag(sq_left_down)
    flag_ru = getflag(sq_right_up)
    flag_rd = getflag(sq_right_down)

    if flag_lu == 1 and flag_ld == 1:
        return "horizontal"
    elif flag_ld == 1 and flag_rd == 1:
        return "vertical"
    elif flag_lu ==1 and flag_rd == 1:
        return "wrongcut"
    elif flag_ld ==1 and flag_ru == 1:
        return "wrongcut"
    else:
        return "god know"

