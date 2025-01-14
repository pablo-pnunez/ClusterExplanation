from json import load
from src.datasets.cluster_explantion.ClusterExplanationBase import *
from src.Common import to_pickle, get_pickle, print_g

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class OnlyFoodClusterExplanation(ClusterExplanationBase):

    def __init__(self, config, restaurant_clustering_data):
        self.RESTAURANT_CLUSTERING_DATA = restaurant_clustering_data
        ClusterExplanationBase.__init__(self, config=config)
    
    def get_data(self, load=["IMG", "IMG_VEC", "N_USR", "V_IMG", "TRAIN", "TEST"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data:
            return dict_data

        # Si no existe, crear
        else:
            # Cargar los ficheros correspondientes (múltiples ciudades)
            RVW, IMG = self.__basic_filtering__()

            # Eliminar reviews sin imágenes
            # RVW = RVW.loc[RVW.restaurantId.isin(RST_NO_IMG)]
            RVW = RVW.loc[RVW.num_images>0]

            # Eliminar los restaurantes que no caen en en los clusters de restaurantes seleccionados y añadir en que cluster caen los otros
            RST_TO_DROP = self.RESTAURANT_CLUSTERING_DATA.loc[self.RESTAURANT_CLUSTERING_DATA.cluster<0, "restaurantId"].values
            RVW = RVW.loc[~RVW.restaurantId.isin(RST_TO_DROP)]
            RVW = RVW.merge(self.RESTAURANT_CLUSTERING_DATA[["restaurantId","cluster"]])

            # Separar en TRAIN y TEST
            ALL_USRS = RVW.userId.unique()
            np.random.seed(self.CONFIG["seed"])
            np.random.shuffle(ALL_USRS)

            USRS_TRAIN = ALL_USRS[:len(ALL_USRS)//2]
            USRS_TEST = ALL_USRS[len(ALL_USRS)//2:]

            RVWS_TRAIN = RVW.loc[RVW.userId.isin(USRS_TRAIN)]
            RSTS_TRAIN = set(RVWS_TRAIN.restaurantId.unique())

            RVWS_TEST = RVW.loc[RVW.userId.isin(USRS_TEST)]
            RSTS_TEST = set(RVWS_TEST.restaurantId.unique())

            # Mover a train los restaurantes que solo aparezcan en test
            MOVE_TO_TRAIN = list(RSTS_TEST-RSTS_TRAIN)
            MOVE_TO_TRAIN_USRS = RVWS_TEST.loc[RVWS_TEST.restaurantId.isin(MOVE_TO_TRAIN)].userId.values
            RVWS_TRAIN = RVWS_TRAIN.append(RVWS_TEST.loc[RVWS_TEST.userId.isin(MOVE_TO_TRAIN_USRS)])
            RVWS_TEST = RVWS_TEST.loc[~RVWS_TEST.userId.isin(MOVE_TO_TRAIN_USRS)]

            # Asegurarse de que en TRAIN, cada restaurante tenga por lo menos "min_imgs_per_rest" imágenes
            IMGS_RST_TRAIN = RVWS_TRAIN.groupby("restaurantId").num_images.sum().reset_index()
            RST_TO_COMPENSATE = IMGS_RST_TRAIN.loc[IMGS_RST_TRAIN.num_images < self.CONFIG["min_imgs_per_rest"]]

            MOVE_TO_TRAIN_USRS = []
            for _, rvw in RST_TO_COMPENSATE.iterrows():
                diff = self.CONFIG["min_imgs_per_rest"] - rvw.num_images
                MOVE_TO_TRAIN_USRS.extend(RVWS_TEST.loc[RVWS_TEST.restaurantId==rvw.restaurantId].userId.tolist())
            MOVE_TO_TRAIN_USRS = list(set(MOVE_TO_TRAIN_USRS))
            
            RVWS_TRAIN = RVWS_TRAIN.append(RVWS_TEST.loc[RVWS_TEST.userId.isin(MOVE_TO_TRAIN_USRS)])
            RVWS_TEST = RVWS_TEST.loc[~RVWS_TEST.userId.isin(MOVE_TO_TRAIN_USRS)]

            # Obtener ID para ONE-HOT de usuarios y restaurantes
            USR_TMP = pd.DataFrame(columns=["real_id", "id_user"])
            REST_TMP = pd.DataFrame(columns=["real_id", "id_restaurant"])

            # Obtener tabla real_id -> id para usuarios
            USR_TMP.real_id = RVWS_TRAIN.sort_values("userId").userId.unique()
            USR_TMP.id_user = range(0, len(USR_TMP))

            # Obtener tabla real_id -> id para restaurantes
            REST_TMP.real_id = RVWS_TRAIN.sort_values("restaurantId").restaurantId.unique()
            REST_TMP.id_restaurant = range(0, len(REST_TMP))

            # Mezclar datos
            RVWS_TRAIN = RVWS_TRAIN.merge(USR_TMP, left_on='userId', right_on='real_id', how='inner')
            RVWS_TRAIN = RVWS_TRAIN.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')
            RVWS_TEST = RVWS_TEST.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='left')

            IMG = IMG.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')
            IMG = IMG.drop(columns=[ "real_id"])

            # Añadir columna identificar imágenes de test
            IMG["test"] = False
            IMG.loc[IMG.reviewId.isin(RVWS_TEST.reviewId), "test"] = True
            RVWS_TRAIN = RVWS_TRAIN[['date', 'images', 'language', 'rating', 'restaurantId', 'reviewId', 'text', 'title', 'url', 'userId', 'num_images', 'id_user', 'id_restaurant','rest_name', 'like']]

            # Separar vectores de imágenes
            IMG = IMG.reset_index(drop=True)
            IMG["id_img"] = range(len(IMG))
            IMG_VEC = np.row_stack(IMG.vector.values)
            IMG = IMG.drop(columns=['vector'])

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "TEST", RVWS_TEST)
            to_pickle(self.DATASET_PATH, "TRAIN", RVWS_TRAIN)
            to_pickle(self.DATASET_PATH, "IMG", IMG)
            to_pickle(self.DATASET_PATH, "IMG_VEC", IMG_VEC)
            to_pickle(self.DATASET_PATH, "USR_TMP", USR_TMP)
            to_pickle(self.DATASET_PATH, "REST_TMP", REST_TMP)
            to_pickle(self.DATASET_PATH, "N_RST", len(REST_TMP))
            to_pickle(self.DATASET_PATH, "N_USR", len(USR_TMP))
            to_pickle(self.DATASET_PATH, "V_IMG", IMG_VEC.shape[1])

            return self.get_dict_data(self.DATASET_PATH, load)
    
    def get_paper_stats(self):
        
        RVW, _ = self.__basic_filtering__()
        RVW = RVW.loc[RVW.num_images>0]
        RST_TO_DROP = self.RESTAURANT_CLUSTERING_DATA.loc[self.RESTAURANT_CLUSTERING_DATA.cluster<0, "restaurantId"].values
        RVW = RVW.loc[~RVW.restaurantId.isin(RST_TO_DROP)]
        RVW = RVW.merge(self.RESTAURANT_CLUSTERING_DATA[["restaurantId","cluster"]])

        for dts in [RVW, self.DATA["TRAIN"]]:

            n_usrs = len(dts.userId .unique())
            n_rsts = len(dts.restaurantId.unique())
            n_imgs = dts.num_images.sum()

            line = " & ".join(map(lambda x: f"{x:,}", [n_usrs, n_rsts, len(dts), n_imgs]))
            print(line+" \\\\")
