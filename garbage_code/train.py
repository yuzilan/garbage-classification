from models import *
from arguments import *
from dataset import *
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.preprocessing.image import ImageDataGenerator

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def Write(write_path, List):
    fileObject = open(write_path, 'w')
    for ip in List:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    print("Write success!")


if __name__ == "__main__":
    FLAGS = argparser()

    model = Model_EfficientNetB5(FLAGS)
    datagen = ImageDataGenerator(
        zca_whitening=True,  # 白化
        zca_epsilon=1e-06,
        rotation_range=90,  # 随机旋转的度数范围
        width_shift_range=0.2,  # 水平平移范围
        height_shift_range=0.2,  # 垂直平移范围
        brightness_range=[0.1, 10],  # 亮度范围
        horizontal_flip=True,  # 垂直翻转
        vertical_flip=True,  # 水平翻转
        validation_split=0.1)

    train_it = datagen.flow_from_directory('../total_datasets/train/', target_size=(456,456), class_mode='categorical', batch_size=4, subset='training')
    val_it = datagen.flow_from_directory('../total_datasets/train/', target_size=(456,456), class_mode='categorical', batch_size=4, subset='validation')
    # test_it = datagen.flow_from_directory('../total_datasets/test/', target_size=(456,456), class_mode='categorical', batch_size=4)

    # 余弦退火
    sample_count = FLAGS.sample_count
    epochs = FLAGS.max_epochs
    warmup_epoch = 5
    batch_size = FLAGS.batch_size
    learning_rate_base = FLAGS.learning_rate
    total_steps = int(epochs * sample_count / batch_size)
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=4e-06,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=30, # 5变30
                                            )

    history = model.fit_generator(
        train_it,
        steps_per_epoch=len(train_it),
        epochs=FLAGS.max_epochs,
        verbose=1,
        validation_data=val_it,
        use_multiprocessing=True,
        validation_steps=6,
        callbacks=[warm_up_lr]
    )

    acc_list = history.history['acc']
    Write('acc_list.txt', acc_list)
    val_acc_list = history.history['val_acc']
    Write('val_acc_list.txt', val_acc_list)
    lost_list = history.history['loss']
    Write('lost_list.txt', lost_list)
    val_lost_list = history.history['val_loss']
    Write('val_lost_list.txt', val_lost_list)

    model.save_weights('small_model_9.h5')  # 大数据训练30在4.h5基础上数据增强

    loss, accuracy = model.evaluate_generator(val_it, steps=len(val_it))
    
    print(loss)
    print(accuracy)
    print('end')

