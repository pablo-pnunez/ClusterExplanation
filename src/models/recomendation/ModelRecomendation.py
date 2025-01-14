# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd
import sys

from src.Common import print_e
from src.models.ModelClass import *

from keras.models import Model, load_model

########################################################################################################################

class ModelRecomendation(ModelClass):

    def __init__(self,config, dataset):
        ModelClass.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        raise NotImplementedError

    def test(self,log2file=False, smp_model=None, position_mode="min", baseline=None):

        if os.path.exists(self.MODEL_PATH+"test.txt") and log2file:
            print_e("The model already exists...")
            #os.remove(self.MODEL_PATH+"test.txt")
            exit()

        #Almacenar la salida en fichero
        if log2file: sys.stdout = open(self.MODEL_PATH+"test_"+position_mode+".txt", "w")

        print("*"*100)
        print(self.CONFIG["model"])
        print("*"*100)

        # Cargar los datos de los usuarios apartados al principio (FINAL_USRS)
        FINAL_USRS = self.DATASET.DATA["TEST"]

        self.MODEL.load_weights(self.MODEL_PATH + "/weights")

        #################################################################################

        ret = []

        for id, r in tqdm(FINAL_USRS.groupby("userId")):

            uid = r["userId"].values[0]
            relevant = r["id_restaurant"].dropna().unique()

            rvw_idxs = r.loc[r.city != self.DATASET.CONFIG["city"]].reviewId.to_list()
            rvw_imgs = np.row_stack(self.DATASET.DATA["IMG_TEST"].loc[self.DATASET.DATA["IMG_TEST"].reviewId.isin(rvw_idxs)].vector.to_list())

            if baseline is None:
                if smp_model is None : preds = self.MODEL.predict(rvw_imgs)
                else: preds = self.MODEL.predict(smp_model.predict(rvw_imgs))

                ordered = np.argsort(-np.sum(preds, axis=0))

            else:
                ordered = baseline

            pos_relevants = {}

            for rt in relevant:
                pos_relevants[rt] =  np.argwhere(ordered==rt).flatten()[0]

            if "min" in position_mode: the_pos = np.min(list(pos_relevants.values()))
            if "max" in position_mode: the_pos = np.max(list(pos_relevants.values()))

            ret.append((uid,the_pos,len(relevant)))

        ret = pd.DataFrame(ret, columns=["id_user","position","n_revs"])

        self.__desglosar_res__(ret)

        # Cerrar fichero de salida
        if log2file: sys.stdout.close()

    def test_baseline(self,position_mode="min"):

        ordered = self.DATASET.DATA["TRAIN"].id_restaurant.value_counts().reset_index().rename(columns={"index": "id_restaurant", "id_restaurant": "n_reviews"}).id_restaurant.values
        self.test(log2file=False, smp_model=None, position_mode=position_mode, baseline=ordered)

    def __desglosar_res__(self, ret):

        # Desglosar resultados por número de restaurantes

        for i in [1, 2, 3, 4]:
            dt = ret.loc[ret.n_revs>=i]
            print("%d\t%d\t%f\t%f\t%f" % (i,len(dt), dt.position.median(),dt.position.mean(),dt.position.std()))