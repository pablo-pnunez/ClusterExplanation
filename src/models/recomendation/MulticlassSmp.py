# -*- coding: utf-8 -*-
from src.models.recomendation.MultilabelSmp import *

import sys

from src.Common import print_e

import keras.backend as K
from keras.layers import Input, BatchNormalization, Dense
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.utils import Sequence, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################################################################################################

class MulticlassSmp(MultilabelSmp):

    def __init__(self,config, dataset):
        MultilabelSmp.__init__(self,config=config, dataset=dataset)

    def get_model(self):

        input = Input(shape=(128,), name="in")
        x = BatchNormalization()(input)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)

        output = Dense(len(self.DATASET.DATA["RST_ADY"]), activation="softmax")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[input], outputs=[output])

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

