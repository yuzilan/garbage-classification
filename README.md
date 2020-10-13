# garbage-classification

- `mean_std.py`：保存所有图像的路径到`imgs_path.txt`，并且运行得到图像的均值和方差保存到`mean_std.txt`

BaseLine改进
1.使用多种模型进行对比实验，ResNet50, SE-ResNet50, Xeception, SE-Xeception, efficientNetB5。

2.使用组归一化（GroupNormalization）代替批量归一化（batch_normalization）-解决当Batch_size过小导致的准确率下降。当batch_size小于16时，BN的error率 逐渐上升，train.py。
```
for i, layer in enumerate(model.layers):
    if "batch_normalization" in layer.name:
        model.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
```
3.NAdam优化器
```
optimizer = Nadam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
```
4.自定义学习率-SGDR余弦退火学习率
```
sample_count = len(train_sequence) * FLAGS.batch_size
epochs = FLAGS.max_epochs
warmup_epoch = 5
batch_size = FLAGS.batch_size
learning_rate_base = FLAGS.learning_rate
total_steps = int(epochs * sample_count / batch_size)
warmup_steps = int(warmup_epoch * sample_count / batch_size)

warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0,
                                        )
```
5.数据增强：随机水平翻转、随机垂直翻转、以一定概率随机旋转90°、180°、270°、随机crop(0-10%)等(详细代码请看aug.py和data_gen.py)
```
def img_aug(self, img):
    data_gen = ImageDataGenerator()
    dic_parameter = {'flip_horizontal': random.choice([True, False]),
                     'flip_vertical': random.choice([True, False]),
                     'theta': random.choice([0, 0, 0, 90, 180, 270])
                    }


    img_aug = data_gen.apply_transform(img, transform_parameters=dic_parameter)
    return img_aug


from imgaug import augmenters as iaa
import imgaug as ia

def augumentor(image):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-10, 10)),
            sometimes(iaa.Crop(percent=(0, 0.1), keep_size=True)),
        ],
        random_order=True
    )


    image_aug = seq.augment_image(image)

    return image_aug
```
6.标签平滑data_gen.py
```
def smooth_labels(y, smooth_factor=0.1):
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y
```
7.数据归一化：得到所有图像的位置信息Save_path.py并计算所有图像的均值和方差mead_std.py
```
normMean = [0.56719673 0.5293289  0.48351972]
normStd = [0.20874391 0.21455203 0.22451781]


img = np.asarray(img, np.float32) / 255.0
mean = [0.56719673, 0.5293289, 0.48351972]
std = [0.20874391, 0.21455203, 0.22451781]
img[..., 0] -= mean[0]
img[..., 1] -= mean[1]
img[..., 2] -= mean[2]
img[..., 0] /= std[0]
img[..., 1] /= std[1]
img[..., 2] /= std[2]
```
