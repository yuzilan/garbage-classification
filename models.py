import os
import shutil
import multiprocessing
import numpy as np
from glob import glob
from arguments import argparser
from tools.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
from tools.Groupnormalization import GroupNormalization
from keras_efficientnets import EfficientNetB5
from keras.applications.inception_v3 import preprocess_input
from keras import backend
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import TensorBoard, Callback
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.layers import Dense
from keras.utils import multi_gpu_model


def Model_EfficientNetB5(FLAGS):
    model = EfficientNetB5(weights=None,
                           include_top=False,
                           input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                           classes=FLAGS.num_classes,
                           pooling=max)
    
    # model.load_weights('/home/work/user-job-dir/src/efficientnet-b5_notop.h5')
    for i, layer in enumerate(model.layers):
        if "batch_normalization" in layer.name:
            model.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)  # activation="linear",activation='softmax'
    optimizer = Nadam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model = Model(input=model.input, output=predictions)
    model.load_weights('../garbage-11-17/small_model_4.h5')

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
