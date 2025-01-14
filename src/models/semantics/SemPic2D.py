# -*- coding: utf-8 -*-

from src.models.semantics.SemPic2 import *

########################################################################################################################

class SemPic2D(SemPic2):

    def __init__(self,config, dataset):
        SemPic2.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        self.DATASET.DATA["N_USR"]= int( self.DATASET.DATA["N_USR"] * self.CONFIG["pctg_usrs"])

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = BatchNormalization()(input)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(2, name="img_emb")(x)
        x = BatchNormalization(name="img_emb_norm")(x)
        output = Dense(self.DATASET.DATA["N_USR"], activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["learning_rate"])
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[self.precision, self.recall, self.f1, "accuracy"])

        return model

    ####################################################################################################################
