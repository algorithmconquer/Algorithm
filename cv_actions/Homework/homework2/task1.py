from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


def generate_vgg16():
    """
    搭建VGG16网络结构
    :return: VGG16网络
    """
    input_shape = (224, 224, 3)
    model = Sequential([
        #(224, 224, 3)==>(224, 224, 64)
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        # (224, 224, 64)==>(112, 112, 64)
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        # (112, 112, 64)==>(112, 112, 128)
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        # (112, 112, 128)==>(56, 56, 128)
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # (56, 56, 128)==>(56, 56, 256)
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        # (56, 56, 128)==>(28, 28, 256)
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # (28, 28, 256)==>(28, 28, 512)
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        # (28, 28, 512)==>(14, 14, 512)
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        # (14, 14, 512)==>(7, 7, 512)
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),

        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax')
    ])

    return model

if __name__ == '__main__':
    model = generate_vgg16()
    model.summary()