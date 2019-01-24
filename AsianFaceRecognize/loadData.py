'''
读取图片，并标定男女Label
'''
from PIL import Image
import os
import numpy as np


def jpg2gif(dirPath, savePath):
    # 列出目标目录下所有文件
    files = os.listdir(dirPath)
    # 遍历整个文件夹
    for f in files:
        if f.endswith('.jpg'):  # 对图片做转换
            pic_path = os.path.join(dirPath, f)  # 路径整合
            # GIF格式保存
            whole_fname = f.split('.')
            main_fname = whole_fname[0]
            save_fname = main_fname + '.gif'
            pic_savePath = os.path.join(savePath, save_fname)
            print(pic_path)
            try:
                img = Image.open(pic_path)
                img_bw = img.convert('L')
                img_bw.save(pic_savePath)
            except IOError:
                print('jpg 2 gif trans error')


def load_data(dirPath):
    data = []
    label = []
    files = os.listdir(dirPath)
    for f in files:
        # 只对gif格式图片做处理
        # 因为jpg格式是压缩的编码格式，损失了很多细节特征
        if f.endswith('.gif'):
            picPath = os.path.join(dirPath, f)
            print("{picPath , " + picPath + "}")
            # 标签标定
            whole_fname = f.split('.')
            main_fname = whole_fname[0]
            print('{main_name , ' + main_fname + "}")
            attr_fname = main_fname.split('_')
            sex_fname = attr_fname[0]
            status_fname = attr_fname[2]

            sex = sex_fname[0:2]
            status = status_fname
            print('sex %s , status %s' % (sex, status))
            label.append((sex, status))

            # 数据集更新
            img = Image.open(picPath)
            imgNpArray = np.asarray(img, dtype='float64')
            # 整个图像拉伸为一维数组
            data.append(np.ndarray.flatten(imgNpArray))

    return data,label


if __name__ == '__main__':
    load_data('./TestPics')
