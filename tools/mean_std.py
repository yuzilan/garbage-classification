import numpy as np
import cv2
import os


def save_path(read_path, write_path_file):
    with open(write_path_file, "w") as f:
        for root, dirs, files in os.walk(read_path, topdown=False):
            for name in files:
                if name.endswith(".jpg"):
                    f.write(os.path.join(root, name) + "\n")
    print("Finsh writing path of JPG in ", write_path_file)


def cal_mean_std():
    read_path = '../total_datasets/train/'
    write_path_file = '../total_datasets/train_paths.txt'
    save_path(read_path, write_path_file)
    means = [0, 0, 0]
    stds = [0, 0, 0]

    index = 1
    num_imgs = 0
    with open(write_path_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = os.path.join(line)
            img = cv2.imread(tmp[:-1])
            img = np.asarray(img)
            img = img.astype(np.float32) / 255.
            for i in range(3):
                means[i] += img[:, :, i].mean()
                stds[i] += img[:, :, i].std()
            index += 1
            num_imgs += 1

    means.reverse()
    stds.reverse()

    means = np.asarray(means) / num_imgs
    stds = np.asarray(stds) / num_imgs

    print("normMean = {}".format(means))
    print("normStd = {}".format(stds))

    return means, stds

