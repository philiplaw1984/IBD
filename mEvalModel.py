import os
import glob
import numpy as np
from keras.preprocessing.image import img_to_array,ImageDataGenerator
import os,glob,itertools,tqdm,cv2,keras
from keras.utils import to_categorical
from config import config
from Build_model import Build_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
config1 = tf.compat.v1.ConfigProto()
config1.gpu_options.per_process_gpu_memory_fraction =0.45 # 占用GPU90%的显存
tf.compat.v1.Session(config=config1)


def get_file(path):
    ends = os.listdir(path)[0].split('.')[-1]
    img_list = glob.glob(os.path.join(path, '*.' + ends))
    return img_list


class PREDICT(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)


    def Evalx(self):

        #channles = 3
        #normal_size =500
        #classNumber = 2

        test_data_path =   'dataset/test'
        #test_data_path =  self.test_data_path
        categories = list(
            map(get_file, list(map(lambda x: os.path.join(test_data_path, x), os.listdir(test_data_path)))))
        data_list = list(itertools.chain.from_iterable(categories))

        images_data, labels_idx, labels = [], [], []


        with_platform = os.name
        for file in tqdm.tqdm(data_list):
            if self.channles == 3:
                img = cv2.imread(file)

                _, w, h = img.shape[::-1]
            elif self.channles == 1:

                img = cv2.imread(file, 0)


            img = cv2.resize(img, (self.normal_size,self.normal_size))
            if with_platform == 'posix':
                label = file.split('/')[-2]
            elif with_platform == 'nt':
                label = file.split('\\')[-2]

            img = img_to_array(img)
            images_data.append(img)
            labels.append(label)

        with open('test_class_idx.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            for label in labels:
                idx = lines.index(label.rstrip())
                labels_idx.append(idx)


        images_data = np.array(images_data, dtype='float32') / 255.0
        labels = to_categorical(np.array(labels_idx), num_classes=self.classNumber)
        # model = Build_model(config).build_model()
        # model.load_weights(r'checkpoints\ResNet50\ResNet50.h5')
        #

        model = Build_model(self.config).build_model()
        if os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'):
            print('weights is loaded')
        else:
            print('weights is not exist')
        model.load_weights(os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'))



        y_predict= model.predict(images_data)
        y_pred = np.argmax(y_predict, axis=1)  # 求出每条数据概率最大的位置序号，即标签序号，从0开始
        y_true = np.array(labels_idx)


        accuracy = round(accuracy_score(y_true, y_pred),4)
        precision = round(precision_score(y_true, y_pred, average='macro'),4)
        recall = round(recall_score(y_true, y_pred, average='macro'),4)
        f1score = round(f1_score(y_true, y_pred, average='macro'),4)

        print('正确率：', accuracy)
        print('精确率：',precision)
        print('召回率：', recall)
        print('F1调和平均值：',f1score)
        return accuracy,precision,recall ,f1score



if __name__ == '__main__':

    modeleval = PREDICT(config)
    accuracy,precision,recall ,f1score = modeleval.Evalx()

    print('done')