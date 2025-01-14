# -*- coding: utf-8 -*-
from src.models.recomendation.Multilabel import *

import sys

from src.Common import print_e

import keras.backend as K
from keras.layers import Input, BatchNormalization, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import Sequence, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################################################################################################

class Multiclass(Multilabel):

    def __init__(self,config, dataset):
        Multilabel.__init__(self,config=config, dataset=dataset)

    def get_model(self):

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = BatchNormalization()(input)
        x = Dense(600, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(400, activation="relu")(x)
        x = BatchNormalization()(x)

        #Es multiclass pero la salida se pone muiltilabel por petición de Antonio
        output = Dense(len(self.DATASET.DATA["RST_ADY"]), activation="softmax")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[input], outputs=[output])

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    ####################################################################################################################
