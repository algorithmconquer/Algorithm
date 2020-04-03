from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model
import keras

## TODO:从pretrained model中提取特征

# 加载pretrained model
def load_vgg16_model(model_path, weights_path):
    """从当前目录下面的 vgg16_exported.json和vgg16_exported.h5两个文件中导入VGG16网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    return model

# 数据处理：bgr==>rgb,中心化
def preprocess_input(x):
    """预处理图像用于网络输入, 将图像由RGB格式转为BGR格式.
       将图像的每一个图像通道减去其均值
    # Arguments
        x: numpy 数组, 4维.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """
    # bgr==>rgb
    x = x[:, :, ::-1]  # or x = x.transpose((2,0,1))如果在OpenCV中处理图像，是BGR的顺序。
    #mean += np.sum(x, axis=(0,1).astype(int))
    x -= x.mean()
    return x

# 将PIL Image转换为array
def load_img_as_np_array(path, target_size):
    """从给定文件加载图像,转换图像大小为给定target_size,返回32位浮点数numpy数组.
    # Arguments
        path: 图像文件路径
        target_size: 元组(图像高度, 图像宽度).
    # Returns
        A PIL Image instance.
    """
    img = pil_image.open(path)
    img = img.resize(target_size, pil_image.NEAREST)
    return np.asarray(img, dtype=K.floatx())

# 从pretrained model中提取特征
def extract_features(directory, model_path, weights_path):
    """提取给定文件夹中所有图像的特征, 将提取的特征保存在文件features.pkl中,
       提取的特征保存在一个dict中, key为文件名(不带.jpg后缀), value为特征值[np.array]
    Args:
        directory: 包含jpg文件的文件夹
    Returns:
        None
    """
    # 加载模型，去掉最后一层
    model = load_vgg16_model(model_path, weights_path)
    # 去掉最后一层
    model.layers.pop()
    model = Model(inputs=model.inputs, output=model.layers[-1].output)
    features = {}
    for fn in listdir(directory):
        imgPath = directory + '/' + fn
        arr = load_img_as_np_array(imgPath, target_size=(224, 224))
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        arr = preprocess_input(arr)
        feature = model.predict(arr, verbose=0)
        id = fn[:-4]
        features[id] = feature
    return features


if __name__ == '__main__':
    # img_path = 'D:/Datasets/cv_datasets/object_detection/3.png'
    # img_path = 'D:/Datasets/cv_datasets/flower_photos/daisy/5547758_eea9edfd54_n.jpg'
    # img = load_img_as_np_array(img_path, (224, 224))
    # img = img[:, :, ::-1]
    # print(img.shape)
    model_path = 'D:/Datasets/vgg16_exported.json'
    weights_path = 'D:/Datasets/vgg16_exported.h5'

    # model = load_vgg16_model(model_path, weights_path)
    # model.layers.pop()
    # model = Model(inputs=model.inputs, output=model.layers[-1].output)
    # input = np.random.random((10, 224, 224, 3))
    # feature = model.predict(input, verbose=0)

    # 提取所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
    directory = 'D:/Datasets/cv_datasets/Flicker8k_Dataset'
    features = extract_features(directory, model_path, weights_path)
    print('提取特征的文件个数：%d' % len(features)) # 8091个features
    print(keras.backend.image_data_format())
    #保存特征到文件
    dump(features, open('features_zw.pkl', 'wb'))



