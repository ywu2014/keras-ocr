import zipfile
import datetime
import string
import math
import os

import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection

import keras_ocr
import tensorflow as tf 
from utils import get_text_generator
import time

train_detector = False
train_recognizer = True

## Generating synthetic data
assert tf.test.is_gpu_available(), 'No GPU is available.'

data_dir = '.'
alphabet = string.digits + string.ascii_letters + '+-×±Φφ. '
fonts = keras_ocr.data_generation.get_fonts(
    alphabet=alphabet,
    cache_dir=data_dir
)

backgrounds = keras_ocr.data_generation.get_backgrounds(cache_dir=data_dir)

myfonts = [
    './myfonts/fangsong_GB2312.ttf', 
    './myfonts/GenShinGothic-Medium.ttf', 
    './myfonts/GenShinGothic-Monospace-Normal.ttf', 
    './myfonts/GenShinGothic-Normal.ttf', 
    './myfonts/GenShinGothic-P-Normal.ttf', 
    './myfonts/GenShinGothic-Regular.ttf', 
    './myfonts/kaiti_GB2312.ttf', 
    './myfonts/msyh.ttf', 
    './myfonts/simhei.ttf', 
    './myfonts/songti.ttf'
]
for myfont in myfonts:
    fonts.append(myfont)

mybackgrounds = [
    './mybackgrounds/bg1.jpg',
    './mybackgrounds/bg2.jpg',
    './mybackgrounds/bg3.jpg',
    './mybackgrounds/bg4.jpg',
    './mybackgrounds/bg5.jpg',
    './mybackgrounds/bg6.jpg',
    './mybackgrounds/bg7.jpg',
    './mybackgrounds/bg8.jpg',
    './mybackgrounds/bg9.jpg',
    './mybackgrounds/bg10.jpg',
]
for mybackground in mybackgrounds:
    backgrounds.append(mybackground)

print("total fonts: ", len(fonts))
print("total backgrounds: ", len(backgrounds))

# text_generator = keras_ocr.data_generation.get_text_generator(alphabet=alphabet)
text_generator = get_text_generator(alphabet=alphabet)
print('The first generated text is:', next(text_generator))

def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test

background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)

image_generators = [
    keras_ocr.data_generation.get_image_generator(
        height=640,
        width=640,
        text_generator=text_generator,
        font_groups={
            alphabet: current_fonts
        },
        backgrounds=current_backgrounds,
        font_size=(60, 120),
        margin=50,
        rotationX=(-0.05, 0.05),
        rotationY=(-0.05, 0.05),
        rotationZ=(-15, 15)
    )  for current_fonts, current_backgrounds in zip(
        font_splits,
        background_splits
    )
]

image, lines = next(image_generators[1])
text = keras_ocr.data_generation.convert_lines_to_paragraph(lines)
print('The first generated validation image (below) contains:', text)
plt.imshow(image)

## Train the detector
if train_detector:
    start_time = time.time()
    print('-'*30)
    print('start training detector!')
    detector = keras_ocr.detection.Detector(weights='clovaai_general')
    detector_batch_size = 1
    epochs=10
    detector_basepath = os.path.join(data_dir, f'detector_epochs-{epochs}_{datetime.datetime.now().isoformat()}')
    detection_train_generator, detection_val_generator, detection_test_generator = [
        detector.get_batch_generator(
            image_generator=image_generator,
            batch_size=detector_batch_size
        ) for image_generator in image_generators
    ]
    detector.model.fit(
        detection_train_generator,
        steps_per_epoch=math.ceil(len(background_splits[0]) / detector_batch_size),
        epochs=epochs,
        workers=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
            tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv'),
            tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5')
        ],
        validation_data=detection_val_generator,
        validation_steps=math.ceil(len(background_splits[1]) / detector_batch_size),
        batch_size=detector_batch_size
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"detector training finished!! total cost: {elapsed_time}s")

if train_recognizer:
    start_time = time.time()
    print('-'*30)
    print('start training recognizer!')
    # TensorFlow 2.x compatibility
    # 这两行代码一定要在model.compile()之前执行 
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.experimental.output_all_intermediates(True)
    # recognizer_alphabet = ''.join(sorted(set(alphabet)))
    recognizer_alphabet = alphabet
    recognizer = keras_ocr.recognition.Recognizer(
        alphabet=recognizer_alphabet,
        weights='kurapan'
    )
    recognizer.compile()
    for layer in recognizer.backbone.layers:
        layer.trainable = False

    max_length = 10
    recognition_image_generators = [
        keras_ocr.data_generation.convert_image_generator_to_recognizer_input(
            image_generator=image_generator,
            max_string_length=min(recognizer.training_model.input_shape[1][1], max_length),
            target_width=recognizer.model.input_shape[2],
            target_height=recognizer.model.input_shape[1],
            margin=1
        ) for image_generator in image_generators
    ]

    # See what the first validation image for recognition training looks like.
    image, text = next(recognition_image_generators[1])
    print('This image contains:', text)
    plt.imshow(image)

    recognition_batch_size = 8
    epochs = 100
    recognizer_basepath = os.path.join(data_dir, f'recognizer_epochs-{epochs}_{datetime.datetime.now().isoformat()}')
    recognition_train_generator, recognition_val_generator, recognition_test_generator = [
        recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=recognition_batch_size,
        lowercase=False
        ) for image_generator in recognition_image_generators
    ]

    recognizer.training_model.fit(
        recognition_train_generator,
        epochs=epochs,
        steps_per_epoch=math.ceil(len(background_splits[0]) / recognition_batch_size),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=25),
            tf.keras.callbacks.CSVLogger(f'{recognizer_basepath}.csv', append=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=f'{recognizer_basepath}.h5')
        ],
        validation_data=recognition_val_generator,
        validation_steps=math.ceil(len(background_splits[1]) / recognition_batch_size),
        workers=0,
        # batch_size=recognition_batch_size
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"recognizer training finished!! total cost: {elapsed_time}s")