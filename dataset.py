import numpy as np
from PIL import Image
from tools.mean_std import cal_mean_std


def image_read(img_path_file):
    # img_path = "../datasets/img_1.jpg"
    img = padding_black(img_path_file)

    img = img.convert('RGB')
    img = np.array(img)
    # print(img.shape)

    # 数据归一化
    # img = np.asarray(img, np.float32) / 255.0
    # mean, std = cal_mean_std()
    # img[..., 0] -= mean[0]
    # img[..., 1] -= mean[1]
    # img[..., 2] -= mean[2]
    # img[..., 0] /= std[0]
    # img[..., 1] /= std[1]
    # img[..., 2] /= std[2]
    # print(img)

    return img  # numpy格式shape:(327, 455, 3)


def padding_black(img_path_file):
    img = Image.open(img_path_file)
    w, h = img.size

    scale = 456. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

    size_fg = img_fg.size
    size_bg = 456

    img_bg = Image.new("RGB", (size_bg, size_bg))

    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))

    img = img_bg
    return img


if __name__ == "__main__":
    img_path = "../smalldatasets/img_1.jpg"
    image_read(img_path)




