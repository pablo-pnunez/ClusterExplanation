# -*- coding: utf-8 -*-
from src.models.recomendation.ModelRecomendation import *

import sys

from src.Common import print_e

import keras.backend as K
from keras.layers import Input, BatchNormalization, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import Sequence, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################################################################################################

class Multilabel(ModelRecomendation):

    def __init__(self,config, dataset):
        ModelRecomendation.__init__(self,config=config, dataset=dataset)

    def train(self, save=False, log2file=False):

        if os.path.exists(self.MODEL_PATH+"train.txt"):
            print_e("The model already exists...")
            return
        else:
            if save: os.makedirs(self.MODEL_PATH, exist_ok=True)

        #Almacenar la salida en fichero
        if log2file: sys.stdout = open(self.MODEL_PATH+"train.txt", "w")

        # Conjuntos de entrenamiento
        train_sequence = self.Sequence(self)

        callbacks = []

        if save:
            os.makedirs(self.LOG_PATH, exist_ok=True)
            mc = ModelCheckpoint(self.MODEL_PATH + "/weights", save_weights_only=True, save_best_only=True, monitor="loss")
            callbacks.append(mc)

        es = EarlyStopping(patience=self.CONFIG["model"]['epochs'], monitor="loss", mode="min")
        callbacks.append(es)


        self.MODEL.fit_generator(train_sequence,
                                 steps_per_epoch=train_sequence.__len__(),
                                 epochs=self.CONFIG["model"]['epochs'],
                                 verbose=2,
                                 workers=1,
                                 callbacks=callbacks,
                                 max_queue_size=25)

        K.clear_session()

        # Cerrar fichero de salida
        if log2file: sys.stdout.close()

    def get_model(self):

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = BatchNormalization()(input)
        x = Dense(600, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(400, activation="relu")(x)
        x = BatchNormalization()(x)

        #Es multiclass pero la salida se pone muiltilabel por petición de Antonio
        output = Dense(len(self.DATASET.DATA["RST_ADY"]), activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[input], outputs=[output])

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    ####################################################################################################################

    class Sequence(Sequence):

        def __init__(self, model):

            self.MODEL = model

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

            x = self.MODEL.DATASET.DATA["IMG_VEC"][btch.id_img.values]
            y = to_categorical(btch.id_restaurant.values, self.N_RESTAURANTS)

            return x, y
