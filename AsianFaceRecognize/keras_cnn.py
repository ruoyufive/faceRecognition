import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from keras.utils import plot_model
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

# 性别识别 SexOrDirction = 0
# 方向识别 SexOrDirction = 1
SexOrDirction = 0

# 验证MyTest中的图像
IsMyTest = True

img_rows = 250  # 图像行数
img_cols = 250  # 图像列数
batchSize = 514  # 批处理数量
epochs = 20  # 训练次数

historyName = './TestPics/Plt/history_2.png'
historyTitle = 'Model accuracy'
modelPlotName = './TestPics/Plt/model_2.png'
modelWeightName = './TestPics/Plt/modelweight_2.h5'

trainPath = './TestPics/TrainPics'
validPath = './TestPics/ValidPics'
testPath = './TestPics/Test'

myTestPath = './TestPics/MyTest'

# 性别字典
sexDict = {'AF': 0, 'AM': 1}
# 人脸朝向字典
headDirDict = {'00F': 90, '30L': 60, '30R': 120,
               '45L': 45, '45R': 135, '60L': 30,
               '60R': 150, '90L': 0, '90R': 180,
               'CO': 90, 'DI': 90, 'FE': 90, 'HA': 90,
               'NE': 90, 'SA': 90, 'SU': 90, 'AN': 90}


def show_train_data(history, hisPath):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(historyTitle)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig(hisPath)
    plt.show()


def show_model(model, modelPlotName=modelPlotName):
    plot_model(model, to_file=modelPlotName)


# 加载训练集中固定文件名的图像
def load_data(dirPath):
    files = os.listdir(dirPath)
    random.shuffle(files)
    picNum = len(files)

    # 文件夹下所有照片个数 * 图片大小
    data = np.empty((picNum, 250 * 250))
    label = np.empty(picNum)
    label.astype(np.byte)
    for index in range(len(files)):
        # 只对gif格式图片做处理
        # 因为jpg格式是压缩的编码格式，损失了很多细节特征
        if files[index].endswith('.gif'):
            picPath = os.path.join(dirPath, files[index])
            # 标签标定
            whole_fname = files[index].split('.')
            main_fname = whole_fname[0]
            attr_fname = main_fname.split('_')
            sex_fname = attr_fname[0]
            status_fname = attr_fname[2]

            sex = sexDict[sex_fname[0:2]]
            status = headDirDict[status_fname]

            # 辨识性别
            if SexOrDirction == 0:
                label[index] = sex
            else:
                # 辨识人的朝向
                label[index] = status

            # 数据集更新
            img = Image.open(picPath)
            imgNpArray = np.asarray(img, dtype='float64') / 255
            # 整个图像拉伸为一维数组
            data[index] = np.ndarray.flatten(imgNpArray)
    data.astype('float32')

    print('label ', label)
    return (data, label)


# 加载拍的照片
def load_data_my(dirPath):
    files = os.listdir(dirPath)
    random.shuffle(files)
    picNum = len(files)

    # 文件夹下所有照片个数 * 图片大小
    data = np.empty((picNum, 250 * 250))
    img_width = 250
    img_height = 250

    for index in range(len(files)):
        picPath = os.path.join(dirPath, files[index])
        # 数据集更新
        img = Image.open(picPath)
        imgConvert = img.convert('L')
        imgResize = imgConvert.resize((img_width, img_height), Image.BILINEAR)
        imgNpArray = np.asarray(imgResize, dtype='float64') / 255
        # 整个图像拉伸为一维数组
        data[index] = np.ndarray.flatten(imgNpArray)
    data.astype('float32')

    return data


def set_model(learnRate=0.005, decay=1e-6, momentum=0.9):
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(5, kernel_size=(3, 3), input_shape=(1, img_rows, img_cols)))
    else:
        model.add(Conv2D(5, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 1)))

    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(10, kernel_size=(3, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1000))  # Full connection

    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if SexOrDirction == 1:
        model.add(Dense(181))
    else:
        model.add(Dense(2))

    model.add(Activation('softmax'))

    sgd = SGD(lr=learnRate, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    show_model(model, modelPlotName)
    return model


def train_model(model, X_train, Y_train, X_val, Y_val, molWeightPath):
    history = model.fit(X_train, Y_train, batch_size=batchSize, epochs=epochs,
                        verbose=1, validation_data=(X_val, Y_val))
    model.save_weights(molWeightPath, overwrite=True)
    show_train_data(history, historyName)
    return model


def test_model(model, X, Y, modelPath):
    model.load_weights(modelPath)
    score = model.evaluate(X, Y, verbose=0)
    print('model summary ', model.summary())
    return score


if __name__ == '__main__':
    # 加载数据集
    (X_train, y_train) = load_data(trainPath)
    (X_valid, y_valid) = load_data(validPath)
    (X_test, y_test) = load_data(testPath)

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)  # 1 为图像像素深度

    print('train samples ', len(X_train))
    print('validate samples ', len(X_valid))
    print('test samples ', len(X_test))

    if SexOrDirction == 0:
        # 因为只识别男女，所以类别为2
        Y_train = np_utils.to_categorical(y_train, 2)
        Y_valid = np_utils.to_categorical(y_valid, 2)
        Y_test = np_utils.to_categorical(y_test, 2)
    else:
        # 辨识朝向
        Y_train = np_utils.to_categorical(y_train, 181)
        Y_valid = np_utils.to_categorical(y_valid, 181)
        Y_test = np_utils.to_categorical(y_test, 181)

    model = set_model()
    train_model(model, X_train, Y_train, X_valid, Y_valid, modelWeightName)
    score = test_model(model, X_test, Y_test, modelWeightName)
    show_model(model, modelPlotName)
    print('score : ', score)

    # 验证训练集中的数据
    model.load_weights(modelWeightName)
    classes = model.predict_classes(X_test, verbose=0)
    accuracy = np.mean(np.equal(y_test, classes))
    print('accuarcy : ', accuracy)
    for index in range(len(y_test)):
        if y_test[index] != classes[index]:
            print(y_test[index], '被错误分成', classes[index])
