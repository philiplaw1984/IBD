#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
import keras
from MODEL import MODEL,ResnetBuilder
import sys
sys.setrecursionlimit(10000)

class Build_model(object):
    def __init__(self,config):
        self.train_data_path = config.train_data_path
        self.checkpoints = config.checkpoints
        self.normal_size = config.normal_size
        self.channles = config.channles
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.classNumber = config.classNumber
        self.model_name = config.model_name
        self.lr = config.lr
        self.config = config
        # self.default_optimizers = config.default_optimizers
        self.data_augmentation = config.data_augmentation
        self.rat = config.rat
        self.cut = config.cut

    def model_confirm(self,choosed_model):
        if choosed_model :
            model = keras.applications.ResNet50(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classNumber)
        #model.summary()
        return model

    def model_compile(self,model):
        adam = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])  # compile之后才会更新权重和模型
        return model

    def build_model(self):
        model = self.model_confirm(self.model_name)
        model = self.model_compile(model)
        return model