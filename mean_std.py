import numpy as np
import cv2
import os


if __name__ == '__main__':
    read_path = "./datasets/train_data/"
    write_path = 'imgs_path.txt'

    with open(write_path, "w") as f:
        for root, dirs, files in os.walk(read_path, topdown=False):
            for name in files:
                if name.endswith(".jpg"):
                    f.write(os.path.join(root, name) + "\n")
    print("Finsh writing path of JPG in imgs_path.txt")
    
    means = [0, 0, 0]
    stds = [0, 0, 0]

    index = 1
    num_imgs = 0
    with open(write_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            print('{}/{}: {}'.format(index, len(lines), line), end='')
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

    with open('mean_std.txt', "w") as f:
        f.write("normMean = {}".format(means) + "\n")
        f.write("normStd = {}".format(stds) + "\n")
    print("Finsh writing mean and std in mean_std.txt")
