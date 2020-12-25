from arguments import argparser
from tools.Groupnormalization import GroupNormalization
from keras_efficientnets import EfficientNetB5
from keras.models import Model
from keras.optimizers import Nadam
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.layers import Dense
from dataset import image_read
import numpy as np
import tensorflow as tf
import cv2
from dataset import *
from tools.mean_std import save_path
from PIL import Image, ImageDraw, ImageFont
from get_test import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if(isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 画矩形框
def draw_rect(img_path_file, label):
    img = cv2.imread(img_path_file)
    img_ = img.copy()
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    xgrad = cv2.Sobel(blurred, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(blurred, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150)
    contours, heriachy = cv2.findContours(edge_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w < 50 or h < 50:
            continue
        num.append(i)
    for i in num:
        if i == 0:
            continue
        contours[0] = np.concatenate((contours[i], contours[0]))
    # 画框
    x, y, w, h = cv2.boundingRect(contours[0])
    img_ = cv2.rectangle(img_, (x, y), (x + w, y + h), (255, 0, 0), 5)
    # cv2.imshow('img', img_)
    # cv2.waitKey(0)

    # dict = {
    #     "0": "Other garbage/disposable snack boxes ",
    #     "1": "Other garbage/stained plastic ",
    #     "2": "Other rubbish/cigarette butts ",
    #     "3": "Other garbage/toothpicks ",
    #     "4": "Other rubbish/broken flowerpots and dishes and bowls ",
    #     "5": "Other garbage/chopsticks ",
    #     "6": "Kitchen waste/leftovers ",
    #     "7": "Kitchen waste/big bones ",
    #     "8": "Kitchen waste/fruit peels ",
    #     "9": "Kitchen waste/fruit pulp ",
    #     "10": "Kitchen waste/tea residue ",
    #     "11": "Kitchen waste/vegetable leaves and root ",
    #     "12": "Kitchen waste/eggshell ",
    #     "13": "Kitchen Waste/fish bones ",
    #     "14": "Recyclable/Charging Treasure ",
    #     "15": "Recyclables/bags ",
    #     "16": "Recyclable/Cosmetic bottle ",
    #     "17": "Recyclable/plastic toys ",
    #     "18": "Recyclable/plastic bowl and bowl ",
    #     "19": "Recyclable/plastic hangers ",
    #     "20": "Recyclable/express bag ",
    #     "21": "Recyclables/plug wires ",
    #     "22": "Recyclables/used clothes ",
    #     "23": "Recyclables/cans ",
    #     "24": "Recyclable/pillow ",
    #     "25": "Recyclable/plush toy ",
    #     "26": "Recyclable/Shampoo Bottle ",
    #     "27": "Recyclables/glasses ",
    #     "28": "Recyclable/leather shoes ",
    #     "29": "Recyclables/chopping boards ",
    #     "30": "Recyclables/cardboard boxes ",
    #     "31": "Recyclable/condiment bottle ",
    #     "32": "Recyclables/bottles ",
    #     "33": "Recyclables/metal food cans ",
    #     "34": "Recyclables/POTS ",
    #     "35": "Recyclables/edible oil drums ",
    #     "36": "Recyclable/soft drink bottle ",
    #     "37": "Hazardous waste/dry battery ",
    #     "38": "Hazardous waste/ointment ",
    #     "39": "Hazardous Waste/Expired Drugs"
    # }

    dict = {
        "0": "其他垃圾/一次性快餐盒",
        "1": "其他垃圾/污损塑料",
        "2": "其他垃圾/烟蒂",
        "3": "其他垃圾/牙签",
        "4": "其他垃圾/破碎花盆及碟碗",
        "5": "其他垃圾/竹筷",
        "6": "厨余垃圾/剩饭剩菜",
        "7": "厨余垃圾/大骨头",
        "8": "厨余垃圾/水果果皮",
        "9": "厨余垃圾/水果果肉",
        "10": "厨余垃圾/茶叶渣",
        "11": "厨余垃圾/菜叶菜根",
        "12": "厨余垃圾/蛋壳",
        "13": "厨余垃圾/鱼骨",
        "14": "可回收物/充电宝",
        "15": "可回收物/包",
        "16": "可回收物/化妆品瓶",
        "17": "可回收物/塑料玩具",
        "18": "可回收物/塑料碗盆",
        "19": "可回收物/塑料衣架",
        "20": "可回收物/快递纸袋",
        "21": "可回收物/插头电线",
        "22": "可回收物/旧衣服",
        "23": "可回收物/易拉罐",
        "24": "可回收物/枕头",
        "25": "可回收物/毛绒玩具",
        "26": "可回收物/洗发水瓶",
        "27": "可回收物/玻璃杯",
        "28": "可回收物/皮鞋",
        "29": "可回收物/砧板",
        "30": "可回收物/纸板箱",
        "31": "可回收物/调料瓶",
        "32": "可回收物/酒瓶",
        "33": "可回收物/金属食品罐",
        "34": "可回收物/锅",
        "35": "可回收物/食用油桶",
        "36": "可回收物/饮料瓶",
        "37": "有害垃圾/干电池",
        "38": "有害垃圾/软膏",
        "39": "有害垃圾/过期药物"
    }

    text = dict[str(label)].split('/')[0]

    # cv2.putText(img_, text, (x, y+30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
    # cv2.imshow('img', img_)
    # cv2.waitKey(0)

    img_ = cv2ImgAddText(img_, text, x, y+30, (0, 0, 0), 20)

    new_name = '../results/' + img_path_file.split('/')[3]
    cv2.imwrite(new_name, img_)
    print("save ok!")


if __name__ == "__main__":
    FLAGS = argparser()
    # 加载模型框架
    model = EfficientNetB5(weights=None,
                           include_top=False,
                           input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                           classes=FLAGS.num_classes,
                           pooling=max)
    for i, layer in enumerate(model.layers):
        if "batch_normalization" in layer.name:
            model.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)  # activation="linear",activation='softmax'
    optimizer = Nadam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model = Model(input=model.input, output=predictions)

    # 加载权重
    model.load_weights('../garbage-12-23/small_model_8.h5')

    save_figure()

    # img_paths_files, labels = get_paths_labels()
    read_path = '../total_datasets/test/'
    write_path_file = '../total_datasets/test_paths.txt'
    save_path(read_path, write_path_file)

    with open(write_path_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img = image_read(line[:-1])
            imgs = []
            imgs.append(img)
            imgs = np.array(imgs)
            y_pred = model.predict(imgs)
            for y in y_pred:
                result = np.argmax(y)
            draw_rect(line[:-1], result)



