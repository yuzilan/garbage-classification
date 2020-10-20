import os
from PIL import Image
import numpy as np
from arguments import argparser
from train import model_fn
from glob import glob
from keras.optimizers import Nadam
from keras import backend
from keras.applications.inception_v3 import preprocess_input
backend.set_image_data_format('channels_last')


def center_img(img, size=None, fill_value=255):
    h, w = img.shape[:2]
    if size is None:
        size = max(h, w)
    shape = (size, size) + img.shape[2:]
    background = np.full(shape, fill_value, np.uint8)
    center_x = (size - w) // 2
    center_y = (size - h) // 2
    background[center_y:center_y + h, center_x:center_x + w] = img
    return background


def preprocess_img(img_path, img_size):
    img = Image.open(img_path)
    resize_scale = img_size / max(img.size[:2])
    img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1]
    img = center_img(img, img_size)
    return img


def load_test_data(FLAGS):
    label_files = glob(os.path.join(FLAGS.test_data_url, '*.txt'))
    test_data = np.ndarray((len(label_files), FLAGS.input_size, FLAGS.input_size, 3), dtype=np.uint8)

    img_names = []
    test_labels = []
    for index, file_path in enumerate(label_files):
        with open(file_path, 'r') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_names.append(line_split[0])
        test_data[index] = preprocess_img(os.path.join(FLAGS.test_data_url, line_split[0]), FLAGS.input_size)
        test_labels.append(int(line_split[1]))
    return img_names, test_data, test_labels


if __name__ == "__main__":
    FLAGS = argparser()

    img_names, test_data, test_labels = load_test_data(FLAGS)
    test_data = preprocess_input(test_data)

    optimizer = Nadam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model = model_fn(FLAGS, 'categorical_crossentropy', optimizer, ['accuracy'])
    model.load_weights('./logs/weights_029_0.8346.h5')
    predictions = model.predict(test_data, verbose=0)

    right_count = 0
    for index, pred in enumerate(predictions):
        predict_label = np.argmax(pred, axis=0)
        test_label = test_labels[index]
        if predict_label == test_label:
            right_count += 1
    accuracy = right_count / len(img_names)
    print('accuracy: %0.4f' % accuracy)
    metric_file_name = os.path.join(FLAGS.train_local, 'metric.json')
    metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % accuracy
    with open(metric_file_name, "w") as f:
        f.write(metric_file_content + '\n')
