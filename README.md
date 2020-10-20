# garbage-classification

1. `argument.py`：Linux系统下命令行选择参数。
2. `datahelper.py`：数据预处理，把数据处理成能输入进网络进行训练的格式。
3. `train.py`：训练并保存模型。
4. `test.py`：对accuracy对模型进行评估。
5. `tools`文件夹：其中每一个py文件都是能提高accuracy的tricks。
6. `mean_std.py`：保存所有图像的路径到`imgs_path.txt`，并且运行得到图像的均值和方差保存到`mean_std.txt`

---

## 数据集
### 垃圾分类标签
共40种，14802张图片（训练集:验证集 = 9:1）。
```
{
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
```

---

## 数据预处理
在`datahelper.py`中：
- 用BaseSequence类构造基础的数据流生成器，把图片处理为可供模型训练的数据格式。
- 用data_flow函数对label标签进行预处理。

---

## 模型介绍
迁移学习：将google已经训练好的EfficientNetB5模型，加上GroupNormalization、GlobalAveragePooling2D、Dropout、Dense等构成新的模型，对我们准备好的数据集进行训练。

---

## 模型评估
评估分类的accuracy。

---

## tricks
`tools`文件夹：
1. 使用组归一化（GroupNormalization）代替批量归一化（batch_normalization），解决当Batch_size过小导致的准确率下降。
2. 使用NAdam优化器。
3. 自定义学习率：SGDR余弦退火学习率。
4. 数据增强：随机水平翻转、随机垂直翻转、以一定概率随机旋转90°、180°、270°、随机crop(0-10%)等。
5. 标签平滑。
6. 数据归一化：得到所有图像的位置信息并计算所有图像的均值和方差（mead_std.py）。

---

## 未来的工作
1 对比模型ResNet50, SE-ResNet50, Xeception, SE-Xeception，分析实验结果并绘图。
2. 进行模型的测试，用给定的测试集展示分类结果，并可视化结果（在图片上进行标注）。
