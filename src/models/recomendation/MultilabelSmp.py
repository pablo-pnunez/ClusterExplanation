# -*- coding: utf-8 -*-
from src.models.recomendation.Multilabel import *

import sys

from src.Common import print_e

import keras.backend as K
from keras.layers import Input, BatchNormalization, Dense
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.utils import Sequence, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################################################################################################

class MultilabelSmp(Multilabel):

    def __init__(self,config, dataset):
        Multilabel.__init__(self,config=config, dataset=dataset)

    def get_model(self):

        input = Input(shape=(128,), name="in")
        x = BatchNormalization()(input)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)

        #Es multiclass pero la salida se pone muiltilabel por petición de Antonio
        output = Dense(len(self.DATASET.DATA["RST_ADY"]), activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[input], outputs=[output])

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def test(self,log2file=False, smp_model=None, position_mode="min", baseline=None):

        smp_pth = "/home/pperez/PycharmProjects/SemPic/models/SemPic2/" + self.DATASET.CONFIG["city"] + "/95cbf6f197f98f366f68d197618e84ba/model.h5"
        img_model = load_model(smp_pth, compile=False)
        img_model = Model(inputs=[img_model.get_layer("in").input], outputs=[img_model.get_layer("img_emb").output])

        Multilabel.test(self,log2file=log2file, smp_model=img_model,position_mode=position_mode, baseline=baseline)

    ####################################################################################################################

    class Sequence(Sequence):

        def __init__(self, model):

            self.MODEL = model

            smp_pth = "/home/pperez/PycharmProjects/SemPic/models/SemPic2/"+self.MODEL.DATASET.CONFIG["city"]+"/95cbf6f197f98f366f68d197618e84ba/model.h5"
            self.img_model = load_model(smp_pth, compile=False)
            self.img_model = Model(inputs=[self.img_model.get_layer("in").input], outputs=[self.img_model.get_layer("img_emb").output])
            self.img_smp = self.img_model.predict(self.MODEL.DATASET.DATA["IMG_VEC"])

            self.TRAIN_DATA = self.MODEL.DATASET.DATA["IMG"].loc[self.MODEL.DATASET.DATA["IMG"].test == 0][["id_img", "id_restaurant"]].sample(frac=1)
            self.N_RESTAURANTS = len(self.TRAIN_DATA.id_restaurant.unique())
            self.BATCH_SIZE = self.MODEL.CONFIG["model"]["batch_size"]

            if len(self.TRAIN_DATA) > self.BATCH_SIZE:
                self.BATCHES = np.array_split(self.TRAIN_DATA, len(self.TRAIN_DATA) // self.BATCH_SIZE)
            else:
                self.BATCHES = np.array_split(self.TRAIN_DATA, 1)

        def __len__(self):
            return len(self.BATCHES)

        def __getitem__(self, idx):

            # print("\nPIDIENDO ITEM %d" % idx)
            btch = self.BATCHES[idx]

            x = self.img_smp[btch.id_img.values]
            y = to_categorical(btch.id_restaurant.values, self.N_RESTAURANTS)

            return x, y
