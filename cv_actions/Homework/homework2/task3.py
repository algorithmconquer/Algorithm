from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from pickle import load
import numpy as np
import util
from pickle import dump

def create_tokenizer():
    """
    根据训练数据集中图像名，和其对应的标题，生成一个tokenizer
    :return:
    """
    path = 'D:/learningDoc/CVSource/CVSource/cv_action_codes/Homework/homework2/'
    train_image_names = util.load_image_names(path + 'task4/Flickr_8k.trainImages.txt')
    train_descriptions = util.load_clean_captions(path + 'task5/descriptions.txt', train_image_names)
    lines = util.to_list(train_descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    #保存特征到文件
    dump(tokenizer, open('tokenizer_zw.pkl', 'wb'))
    return tokenizer

def testTokenizer():
    tokenizer = Tokenizer()
    lines = ['this is good', 'that is a cat']
    tokenizer.fit_on_texts(lines) # {'is': 1, 'this': 2, 'good': 3, 'that': 4, 'a': 5, 'cat': 6}
    results = tokenizer.texts_to_sequences(['cat is good'])
    print(results[0])


def create_input_data_for_one_image(seq, photo_feature, max_length, vocab_size):
    """

    :param seq: 图片的标题（已经将英文单词转换为整数）序列
    :param photo_feature: 图片的特征numpy数组
    :param max_length:训练数据集中最长的标题长度
    :param vocab_size:训练数据集中图像标题的单词数量
    :return:tuple
            第一个元素为list, list的元素为图像的特征==图像输入
            第二个元素为list, list的元素为图像标题的前缀==文字输入
            第三个元素为list, list的元素为图像标题的下一个单词(根据图像特征和标题的前缀产生)的one-hot编码==输出
    """
    # input1, input2, output长度一样
    # seq = [2, 660, 6, 229, 3]时，长度为5；
    input1 = list() #
    input2 = list() #
    output = list()
    print(seq)
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        input2.append(in_seq)
        output.append(out_seq)
        input1.append(photo_feature)
    return input1, input2, output




def create_input_data(tokenizer, max_length, descriptions, photos_features, vocab_size):
    """
    从输入的图片标题list和图片特征构造LSTM的一组输入

    Args:
    :param tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
    :param max_length: 训练数据集中最长的标题的长度
    :param descriptions: dict, key 为图像的名(不带.jpg后缀), value 为list, 包含一个图像的几个不同的描述
    :param photos_features:  dict, key 为图像的名(不带.jpg后缀), value 为numpy array 图像的特征
    :param vocab_size: 训练集中表的单词数量
    :return: tuple:
            第一个元素为 numpy array, 元素为图像的特征, 它本身也是 numpy.array
            第二个元素为 numpy array, 元素为图像标题的前缀, 它自身也是 numpy.array
            第三个元素为 numpy array, 元素为图像标题的下一个单词(根据图像特征和标题的前缀产生) 也为numpy.array

    Examples:
        from pickle import load
        tokenizer = load(open('tokenizer.pkl', 'rb'))
        max_length = 6
        descriptions = {'1235345':['startseq one bird on tree endseq', "startseq red bird on tree endseq"],
                        '1234546':['startseq one boy play water endseq', "startseq one boy run across water endseq"]}
        photo_features = {'1235345':[ 0.434,  0.534,  0.212,  0.98 ],
                          '1234546':[ 0.534,  0.634,  0.712,  0.28 ]}
        vocab_size = 7378
        print(create_input_data(tokenizer, max_length, descriptions, photo_features, vocab_size))
(array([[ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ]]),
array([[  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59, 254],
       [  0,   0,   2,  59, 254,   6],
       [  0,   2,  59, 254,   6, 134],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  26],
       [  0,   0,   0,   2,  26, 254],
       [  0,   0,   2,  26, 254,   6],
       [  0,   2,  26, 254,   6, 134],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59,  16],
       [  0,   0,   2,  59,  16,  82],
       [  0,   2,  59,  16,  82,  24],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59,  16],
       [  0,   0,   2,  59,  16, 165],
       [  0,   2,  59,  16, 165, 127],
       [  2,  59,  16, 165, 127,  24]]),
array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]]))
    """
    pass

if __name__ == '__main__':
    # testTokenizer()
    # create_tokenizer()
    photo_feature = np.array([0.345, 0.57, 0.003, 0.987])
    tokenizer = load(open('tokenizer_zw.pkl', 'rb'))
    vocab_size = len(tokenizer.word_index)
    desc = 'startseq cat on table endseq'
    seq = tokenizer.texts_to_sequences([desc])[0] # [2, 660, 6, 229, 3]
    path = 'D:/learningDoc/CVSource/CVSource/cv_action_codes/Homework/homework2/'
    train_image_names = util.load_image_names(path + 'task4/Flickr_8k.trainImages.txt')
    captions = util.load_clean_captions(path + 'task5/descriptions.txt', train_image_names)
    max_length = util.get_max_length(captions)
    # input1, input2, output = create_input_data_for_one_image(seq, photo_feature, 6, 661)
    # print(input1)
    # print(input2)
    # print(output)
    input1, input2, output = create_input_data_for_one_image(seq, photo_feature, max_length, vocab_size)