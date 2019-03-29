# -*- encoding: utf-8 -*-
import os
import cv2
import json
import numpy as np
import tensorflow as tf
from app.util import get_batch, get_num_examples, readFile

# model's path
MODEL_COLORS_PATH = '../result/color_model/'
MODEL_NC_PATH = '../result/recognize_num_char_model/'
MODEL_CH_PATH = '../result/recognize_chinese_model/'
MODEL_LASTCH_PATH = '../result/recognize_lastch_model/'
MODE_NAME_COLORS = "model_colors.ckpt"
MODE_NAME_CHARS_NC = "model_chars_nc.ckpt"
MODE_NAME_CHARS_CH = "model_chars_ch.ckpt"
MODE_NAME_LASTCH_CH = "model_lastch_ch.ckpt"

# const param configure
MIN_W_H_RATE = 1.4
MAX_W_H_RATE = 3.2

# parameter setting
INPUT_NODE_CHARS = 448
INPUT_NODE_COLORS = 800
OUTPUT_NODE_CHARS_NC = 34
OUTPUT_NODE_CHARS_CH = 31
OUTPUT_NODE_LASTCH_CH = 4
OUTPUT_NODE_COLORS = 5
IMAGE_W_C = 100
IMAGE_H_C = 32
IMAGE_W_P = 16
IMAGE_H_P = 28

NUM_CHANNELS_CHARS = 1
NUM_LABELS_CHARS = 65
NUM_LABELS_LASTCHARS = 4
NUM_CHANNELS_COLORS = 3
NUM_LABELS_COLORS = 5

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

# parameter configure
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99


# MODE_SAVE_PATH = "./plate_model/"

def recognise(img_color, img_char_list, img_type=0):
    # print(np.array(img_color).shape)
    # print(np.array(img_char_list).shape)
    _is_train_file()
    # print(np.array(img_color).shape)
    # print(len(img_color))
    # # print(np.array(img_char_list).shape)
    # print(len(img_char_list))
    for img_char in img_char_list:
        # print(len(img_char))
        if len(img_char) == 1:
            continue
        for chars in img_char:
            # cv2.imshow('img', chars)
            # cv2.waitKey(0)
            # print(np.array(chars).shape)
            pass
    pre_results_colors, img_color_lists = _pretreatment_color(img_color)
    results_colors = _predict(pre_results_colors, 1)
    # results_colors = _predict(img_color, 1)
    results_lastch_ch = []
    img_char_lists_nc, img_char_lists_ch, img_char_lists_lastch, img_len_lists = _pretreatment_char(img_char_list, img_type)

    if img_type == 1:
        results_lastch_ch = _predict(img_char_lists_lastch, 4)
    results_chars_nc = _predict(img_char_lists_nc, 2)
    results_chars_ch = _predict(img_char_lists_ch, 3)
    # results = generate_result(results_chars_ch, results_chars_nc, results_colors, img_len_lists, img_color_lists)
    results = generate_result(results_chars_ch, results_chars_nc, results_lastch_ch, results_colors, img_len_lists, img_type)
    # print(img_len_lists)
    # print(results)
    return results


def _pretreatment_color(img_color):
    img_color_lists = []
    pre_res_colors = []
    for img_c in img_color:
        # print(len(img_c))
        if len(img_c) == 32:
            img_color_lists.append(1)
            pre_res_colors.append(img_c)
        else:
            img_color_lists.append(0)
    return pre_res_colors, img_color_lists


def _pretreatment_char(img_char_list, img_type):
    img_char_lists_nc = []
    img_char_lists_ch = []
    img_len_lists = []
    img_char_list_lastch = []
    # print(np.array(img_char_list).shape)
    for img_char in img_char_list:
        # print(np.array(img_char).shape)
        if len(img_char) == 7:
            img_len_lists.append(7)
        elif len(img_char) == 8:
            img_len_lists.append(8)
        elif len(img_char) == 1:
            img_len_lists.append(1)
            continue
        flag = True
        i = 0
        for chars in img_char:
            if flag:
                img_char_lists_ch.append(chars)
                flag = False
            else:
                if (img_type == 1) and (i == (len(img_char) - 1)):
                    img_char_list_lastch.append(chars)
                    # print(chars)
                else:
                    img_char_lists_nc.append(chars)
                    # print(np.array(chars).shape)
            i += 1
    # pre_images = []
    # for filename in filenames:
    #     # img = load_image_data(filename, (50, 16))
    #     img_sq = np.squeeze(np.array(filename) / 255.0)
    #     pre_images.append
    # (img_sq)

    return img_char_lists_nc, img_char_lists_ch, img_char_list_lastch, img_len_lists


def generate_result(results_chars_ch, results_chars_nc, results_lastch_ch, results_colors, img_len_lists, img_type):
    chars = {'zh_cuan': '川', 'zh_e': '鄂', 'zh_gan': '赣', 'zh_gan1': '甘', 'zh_gui': '贵', 'zh_gui1': '桂', 'zh_hei': '黑',
             'zh_hu': '沪', 'zh_ji': '冀', 'zh_ji1': '吉', 'zh_jin': '津', 'zh_jin1': '晋', 'zh_jing': '京',
             'zh_liao': '辽', 'zh_lu': '鲁', 'zh_meng': '蒙', 'zh_min': '闽', 'zh_ning': '宁', 'zh_qing': '青',
             'zh_qiong': '琼', 'zh_shan': '陕', 'zh_su': '苏', 'zh_wan': '皖', 'zh_xiang': '湘', 'zh_xin': '新',
             'zh_yu': '豫', 'zh_yu1': '渝', 'zh_yue': '粤', 'zh_yun': '云', 'zh_zang': '藏', 'zh_zhe': '浙'}
    colors = {'blue': '蓝', 'white': '白', 'black': '黑', 'yellow': '黄', 'green': '绿'}
    last_chs = {'zh_ao': '澳', 'zh_gang': '港', 'zh_jing': '警', 'zh_xue': '学'}
    res = ""
    res_list = []
    i = 0
    # 图片张数标记
    j = 0
    # 有结果的图片标记
    k = 0
    char_flag = 0
    # print('length of results_last_ch', len(results_lastch_ch))
    # print('length of results_chars_ch', len(results_chars_ch))
    for img_len in img_len_lists:
        if img_len == 1:
            res = ("unknown", "unknown")
            res_list.append(res)
            res = ''
            continue
        else:
            # for chars_nc in results_chars_nc:
            while i < img_len:
                chars_nc = results_chars_nc[char_flag]
                if 0 == i:
                    chars_ch = results_chars_ch[j]
                    res = res + chars[chars_ch]
                    # i = i + 1

                res = (res + chars_nc).upper()
                i = i + 1
                char_flag += 1
                if (img_type == 1) and ((img_len - 2) == i):
                    last_ch = results_lastch_ch[j]
                    res = res + last_chs[last_ch]
                    i += 1
                if (img_len - 1) == i:
                    i = 0
                    # if img_color_lists[j] == 1:
                    #     color = results_colors[k]
                    #     k = k + 1
                    # else:
                    #     color = "unknown"
                    if img_len == 8:
                        if res[1:3]=='WJ':
                            ecch = res[1:3]+res[0]
                            res = res.replace(res[0:3], ecch)
                    color = results_colors[k]
                    k = k + 1
                    j = j + 1
                    if color in colors:
                        res = (res, colors[color])
                    else:
                        res = (res, color)
                    res_list.append(res)
                    res = ''
                    break

    return res_list


def _is_train_file(retrain=0):
    # 是否从新训练模型
    if retrain == 1:
        for file in os.listdir(MODEL_COLORS_PATH):
            path_file = os.path.join(MODEL_COLORS_PATH, file)
            os.remove(path_file)
        print(MODEL_COLORS_PATH + "file is deleted")
        for file in os.listdir(MODEL_NC_PATH):
            path_file = os.path.join(MODEL_NC_PATH, file)
            os.remove(path_file)
        print(MODEL_NC_PATH + "file is deleted")
        for file in os.listdir(MODEL_CH_PATH):
            path_file = os.path.join(MODEL_CH_PATH, file)
            os.remove(path_file)
        print(MODEL_CH_PATH + "file is deleted")
        for file in os.listdir(MODEL_LASTCH_PATH):
            path_file = os.path.join(MODEL_LASTCH_PATH, file)
            os.remove(path_file)
        print(MODEL_LASTCH_PATH + "file is deleted")
        os.removedirs(MODEL_COLORS_PATH)
        os.removedirs(MODEL_NC_PATH)
        os.removedirs(MODEL_CH_PATH)
        print("dir is deleted")
    # 判断路径是否存在
    if os.path.exists("../result/"):
        print("result is exsit")
    else:
        os.mkdir("../result/")
        print("result is established")
    if os.path.exists(MODEL_COLORS_PATH):
        print("colors_path is exist")
    else:
        os.mkdir(MODEL_COLORS_PATH)
        print("color is established")
    if os.path.exists(MODEL_NC_PATH):
        print("nc_path is exist")
    else:
        os.mkdir(MODEL_NC_PATH)
        print("nc is established")
    if os.path.exists(MODEL_CH_PATH):
        print("ch_path is exist")
    else:
        os.mkdir(MODEL_CH_PATH)
        print("ch is established")
    if os.path.exists(MODEL_LASTCH_PATH):
        print("lastch_path is exist")
    else:
        os.mkdir(MODEL_LASTCH_PATH)
        print("lastch is established")
    if 0 == len(os.listdir(MODEL_COLORS_PATH)):
        print("The following train data is color's model")
        image_lists, n_classes = readFile(1)
        _train(image_lists, n_classes, 1)
        tf.reset_default_graph()
    if 0 == len(os.listdir(MODEL_NC_PATH)):
        print("The following train data is num&char's model")
        image_lists, n_classes = readFile(2)
        _train(image_lists, n_classes, 2)
        tf.reset_default_graph()
    if 0 == len(os.listdir(MODEL_CH_PATH)):
        print("The following train data is chinese's model")
        image_lists, n_classes = readFile(3)
        _train(image_lists, n_classes, 3)
    if 0 == len(os.listdir(MODEL_LASTCH_PATH)):
        print("The following train data is lastch's model")
        image_lists, n_classes = readFile(4)
        _train(image_lists, n_classes, 4)


def _inference(input_tensor, train, regularizer, file_type):
    if 1 == file_type:
        num_channels = NUM_CHANNELS_COLORS
        num_lables = NUM_LABELS_COLORS
    else:
        if 4 == file_type:
            num_channels = NUM_CHANNELS_CHARS
            num_lables = NUM_LABELS_LASTCHARS
        else:
            num_channels = NUM_CHANNELS_CHARS
            num_lables = NUM_LABELS_CHARS
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, num_channels, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP],
                                       initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        pool_shape = pool2.get_shape().as_list()
        # pool_shape[0] is number of batch
        # pool_shape[1] is length,pool_shape[2] is width,pool_shape[3] is depth
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    # reshaped = tf.reshape(pool2,[-1,nodes])

    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE],
                                     initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, num_lables],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [num_lables],
                                     initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


# file_type=1表示颜色；=2表示分割图
def _train(image_lists, n_classes, file_type):
    if 1 == file_type:
        image_width = IMAGE_W_C
        image_height = IMAGE_H_C
        num_channels = NUM_CHANNELS_COLORS
        output_nodes = OUTPUT_NODE_COLORS
        model_path = MODEL_COLORS_PATH
        model_name = MODE_NAME_COLORS
    elif 2 == file_type:
        image_width = IMAGE_W_P
        image_height = IMAGE_H_P
        num_channels = NUM_CHANNELS_CHARS
        output_nodes = OUTPUT_NODE_CHARS_NC
        model_path = MODEL_NC_PATH
        model_name = MODE_NAME_CHARS_NC
    elif 3 == file_type:
        image_width = IMAGE_W_P
        image_height = IMAGE_H_P
        num_channels = NUM_CHANNELS_CHARS
        output_nodes = OUTPUT_NODE_CHARS_CH
        model_path = MODEL_CH_PATH
        model_name = MODE_NAME_CHARS_CH
    elif 4 == file_type:
        image_width = IMAGE_W_P
        image_height = IMAGE_H_P
        num_channels = NUM_CHANNELS_CHARS
        output_nodes = OUTPUT_NODE_LASTCH_CH
        model_path = MODEL_LASTCH_PATH
        model_name = MODE_NAME_LASTCH_CH

    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,  # None when inference.layer4-pool2 use reshaped = tf.reshape(pool2,[-1,nodes])
        image_width,
        image_height,
        num_channels],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [
        None,
        output_nodes],
                        name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # when trianing,the second param is True.when testing,it is False
    y = _inference(x, True, regularizer, file_type)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        get_num_examples(image_lists, 'training') / BATCH_SIZE,
        LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(TRAINING_STEPS):
            xs, ys = get_batch(n_classes, image_lists, BATCH_SIZE, 'training', file_type)
            reshape_xs = np.reshape(xs, (
                BATCH_SIZE,
                image_width,
                image_height,
                num_channels))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y_: ys})
            if (i + 1) % 100 == 0:
                vx, vy = get_batch(n_classes, image_lists, BATCH_SIZE, 'validation', file_type)
                reshape_vx = np.reshape(vx, (
                    BATCH_SIZE,
                    image_width,
                    image_height,
                    num_channels))
                _, v_loss = sess.run([train_op, loss], feed_dict={x: reshape_vx, y_: vy})
                print("After %d training step(s),loss on training batch is %g and on validation is %g" % (
                    (i + 1), loss_value, v_loss))

                saver.save(sess, os.path.join(model_path, model_name), global_step=global_step)


def _predict(img_char_list, file_type):
    if 1 == file_type:
        image_width = IMAGE_W_C
        image_height = IMAGE_H_C
        num_channels = NUM_CHANNELS_COLORS
        model_path = MODEL_COLORS_PATH
        lable_name = "../src/labels_color.json"
        # lenth = 1

    elif 2 == file_type:
        image_width = IMAGE_W_P
        image_height = IMAGE_H_P
        num_channels = NUM_CHANNELS_CHARS
        model_path = MODEL_NC_PATH
        lable_name = "../src/labels_chars_nc.json"
        # lenth = 6

    elif 3 == file_type:
        image_width = IMAGE_W_P
        image_height = IMAGE_H_P
        num_channels = NUM_CHANNELS_CHARS
        model_path = MODEL_CH_PATH
        lable_name = "../src/labels_chars_ch.json"
        # lenth = 1

    elif 4 == file_type:
        image_width = IMAGE_W_P
        image_height = IMAGE_H_P
        num_channels = NUM_CHANNELS_CHARS
        model_path = MODEL_LASTCH_PATH
        lable_name = "../src/labels_lastch.json"


    num_examples = len(img_char_list)
    with open(lable_name, "r") as f:
        labels = json.load(f)
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            num_examples,
            image_width,
            image_height,
            num_channels],
                           name='x-input')

        reshape_xs = np.reshape(img_char_list, (
            num_examples,
            image_width,
            image_height,
            num_channels))

        dict_feed = {x: reshape_xs}
        # validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}

        y = _inference(x, False, None, file_type)
        y_pre = tf.argmax(tf.nn.softmax(y), 1)
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                prediction = sess.run([y_pre], feed_dict=dict_feed)
                # print prediction
                str_list = []
                i = 0
                for item in prediction[0]:
                    str_list.append(labels[item])
                    i = i + 1
                # print("the predict result:   ", str_list)
            else:
                print('No checkpoint file found')
    return str_list
