import os.path
import glob
import random
import numpy as np
from PIL import Image
import cv2
from numpy import fft

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
IMAGE_W_C = 50
IMAGE_H_C = 16
IMAGE_W_P = 16
IMAGE_H_P = 28

INPUT_DATA_C = '../src/colors/'
INPUT_DATA_NC = '../src/chars_nc/'
INPUT_DATA_CH = '../src/chars_ch/'
INPUT_DATA_LASTCH = '../src/char_lastch/'

def _create_image_lists(testing_percentage, validation_percentage, input_data):
    result = {}
    sub_dirs = [x[0] for x in os.walk(input_data)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['png', 'jpg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(input_data, dir_name, '*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if len(training_images)== 0 and len(testing_images)== 0 and len(validation_images)== 0:
                validation_images.append(base_name)
                testing_images.append(base_name)
                training_images.append(base_name)
                continue
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {

            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }

    return result


def _get_image_path(img_lists, label_name, index, category, input_data):
    label_lists = img_lists[label_name]
    category_list = label_lists[category]
    if 0 != len(category_list):
        mod_index = index % len(category_list)
    else:
        mod_index = 0
        # print("index = 0")
        # print(len(category_list))
        # print(label_name)
        # print(len(label_lists))
    # print(mod_index)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(input_data, sub_dir, base_name)

    return full_path


def load_image_data(image_path, file_type, size=(16, 23)):
    if file_type == 1:
        img = Image.open(image_path)
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image, (50, 16))
    else:
        image = Image.open(image_path).convert(mode='L')
        image_resized = image.resize(size, Image.BICUBIC)
    return image_resized


def _get_image_tensor(img_list, label_name, index, category, input_data, file_type):
    if file_type == 1:
        image_w = IMAGE_W_C
        image_h = IMAGE_H_C
    else:
        image_w = IMAGE_W_P
        image_h = IMAGE_H_P
    img_path = _get_image_path(img_list, label_name, index, category, input_data)
    img = load_image_data(img_path, file_type, (image_w, image_h))

    return np.squeeze(np.array(img)/255.0)


def get_batch(n_classes, image_lists, how_many, category, file_type):

    if file_type == 1:
        input_data = INPUT_DATA_C
    elif file_type == 2:
        input_data = INPUT_DATA_NC
    elif file_type == 3:
        input_data = INPUT_DATA_CH
    elif file_type == 4:
        input_data = INPUT_DATA_LASTCH
    # image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    # n_classes = len(image_lists)
    imgs = []
    labels = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        img = _get_image_tensor(image_lists, label_name, image_index, category, input_data, file_type)
        imgs.append(img)
        label = np.zeros(n_classes, dtype=np.float32)
        label[label_index] = 1.0
        labels.append(label)

    return imgs, labels


def get_num_examples(image_lists, data_type):
    num_examples = 0
    # image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    for k in image_lists:
        num_examples += len(image_lists[k][data_type])
    return num_examples


# file_type=1表示颜色；=2表示分割图
def readFile(file_type):
    if file_type == 1:
        input_data = INPUT_DATA_C
    elif file_type == 2:
        input_data = INPUT_DATA_NC
    elif file_type == 3:
        input_data = INPUT_DATA_CH
    elif file_type == 4:
        input_data = INPUT_DATA_LASTCH
    image_lists = _create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE, input_data)
    n_classes = len(image_lists)
    return image_lists, n_classes



def inverse(input, PSF, eps):       # 逆滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps #噪声功率，这是已知的，考虑epsilon
    result = fft.ifft2(input_fft / PSF_fft) #计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result


def main():
    # print("the file_type is 1\n")
    # image_lists, n_classes = readFile(1)
    # num_exams = get_num_examples(image_lists, 'validation')
    # print("validatin", num_exams)
    # imgs, labels = get_batch(n_classes, image_lists, get_num_examples(image_lists, "validation"), 'validation', 1)
    # print(len(imgs), len(labels))
    print("\nthe file_type is 2\n")
    image_lists, n_classes = readFile(2)
    num_exams = get_num_examples(image_lists, 'training')
    print("training", num_exams)
    imgs, labels = get_batch(n_classes, image_lists, get_num_examples(image_lists, "training"), 'training', 2)
    print(len(imgs), len(labels))

if __name__ == '__main__':
    main()


