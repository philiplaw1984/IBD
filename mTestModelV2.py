from config import config
import sys,copy,shutil
import cv2
import os,time
import keras

from keras.preprocessing.image import load_img,img_to_array
from keras.applications.resnet50 import preprocess_input
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from Build_model import Build_model
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
config1 = tf.compat.v1.ConfigProto()
config1.gpu_options.per_process_gpu_memory_fraction =0.45 # 占用GPU90%的显存
tf.compat.v1.Session(config=config1)

#
# def grad_cam(model, x, category_index, layer_name):
#     """
#     Args:
#        model: model
#        x: image input
#        category_index: category index
#        layer_name: last convolution layer name
#     """
#     # 取得目标分类的CNN输出值，也就是loss
#     class_output = model.output[:, category_index]
#     print(class_output)
#     # 取得想要算出梯度的层的输出
#     convolution_output = model.get_layer(layer_name).output
#     print(convolution_output)
#     # 利用gradients函数，算出梯度公式
#     grads = K.gradients(class_output, convolution_output)[0]
#     print(grads)
#     # 定义计算函数（tensorflow的常见做法，与一般开发语言不同，先定义计算逻辑图，之后一起计算。）
#     gradient_function = K.function([model.input], [convolution_output, grads])
#     print("开始问题")
#     print(x.shape)
#
#     # 根据实际的输入图像得出梯度张量(返回是一个tensor张量，VGG16 是一个7X7X512的三维张量)
#     output, grads_val = gradient_function([x])
#     print("输出")
#     print(output)
#     print("梯度")
#     print(grads_val)
#     output, grads_val = output[0], grads_val[0]
#
#     # 取得所有梯度的平均值(维度降低：7X7X512 -> 512)
#     weights = np.mean(grads_val, axis=(0, 1))
#     # 把所有平面的平均梯度乘到最后卷积层(vgg16最后一层是池化层)上，得到一个影响输出的梯度权重图
#     cam = np.dot(output, weights)
#
#     # 把梯度权重图RGB化
#     cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
#     cam = np.maximum(cam, 0)
#     heatmap = cam / np.max(cam)
#
#     # Return to BGR [0..255] from the preprocessed image
#     image_rgb = x[0, :]
#     image_rgb -= np.min(image_rgb)
#     image_rgb = np.minimum(image_rgb, 255)
#
#     cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#     cam = np.float32(cam) + np.float32(image_rgb)
#     cam = 255 * cam / np.max(cam)
#     return np.uint8(cam), heatmap
#


class PREDICT(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

    def classes_id(self):
        with open('train_class_idx.txt','r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        return lines

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path

    def Predict(self,picpt):

        model = Build_model(self.config).build_model()
        if os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'):
            print('weights is loaded')
        else:
            print('weights is not exist')
        model.load_weights(os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'))

        if (self.channles == 3):
            img = cv2.resize(cv2.imread(picpt),(self.normal_size, self.normal_size))
        elif (self.channles == 1):
            img = cv2.resize(cv2.imread(picpt,0), (self.normal_size, self.normal_size))
        imgx=img.copy()

        img = np.array([img_to_array(img)],dtype='float')/255.0
        pred = model.predict(img).tolist()[0]
        labelid = pred.index(max(pred))
        label = self.classes_id()[pred.index(max(pred))]
        confidence =round(max(pred),4)
        print('predict label     is: ',label)
        print('predict confidect is: ',confidence)
        return labelid,label, confidence,imgx


    def Predict_Grad_CAM(self,picpt):

        model = Build_model(self.config).build_model()
        if os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'):
            print('weights is loaded')
        else:
            print('weights is not exist')
        model.load_weights(os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'))

        if (self.channles == 3):
            img = cv2.resize(cv2.imread(picpt),(self.normal_size, self.normal_size))
        elif (self.channles == 1):
            img = cv2.resize(cv2.imread(picpt,0), (self.normal_size, self.normal_size))
        imgx=img.copy()
        img = np.array([img_to_array(img)],dtype='float')/255.0
        pred = model.predict(img).tolist()[0]
        class_idx = pred.index(max(pred))
        label = self.classes_id()[pred.index(max(pred))]
        confidence =round(max(pred),4)
        print('predict label     is: ', label)
        print('predict confidect is: ', confidence)

        class_output = model.output[:, class_idx]
        # 需根据自己情况修改2. 把block5_conv3改成自己模型最后一层卷积层的名字
        last_conv_layer = model.get_layer("activation_49")
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

        image = load_img(picpt, target_size=(500, 500))
        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        pooled_grads_value, conv_layer_output_value = iterate([x])
        ##需根据自己情况修改3. 512是我最后一层卷基层的通道数，根据自己情况修改
        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        img = cv2.imread(picpt)
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
        # img = img_to_array(image)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)



        return class_idx,label, confidence,imgx,heatmap,superimposed_img




if __name__=='__main__':

    picpt = r'testpics/1.jpg'
    predict = PREDICT(config)
    labelid,label, confidence,imgx,heatmap,superimposed_img = predict.Predict_Grad_CAM(picpt)

    outputpicpt = r'Output'
    if not os.path.isdir(outputpicpt):
        os.makedirs(outputpicpt)
    picname = os.path.basename(picpt)
    heatmap_name = 'heatmap_'+picname
    heatmap_name_pt =  outputpicpt+'\\'+heatmap_name
    cv2.imwrite(heatmap_name_pt,heatmap)

    gradCAM_name = 'gradcam_' + picname
    gradCAM_name_pt = outputpicpt + '\\' +  gradCAM_name
    cv2.imwrite(gradCAM_name_pt, superimposed_img)


    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(imgx,cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title('Input Img')
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title('Heatmap Img')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img,cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title('GradCAM Img')
    allpt = outputpicpt + '\\' +  'all_' + picname
    plt.savefig(allpt)

    plt.show()








