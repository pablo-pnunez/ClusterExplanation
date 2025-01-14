from src.datasets.cluster_explantion.ClusterExplanationBase import *
from src.Common import to_pickle

from scipy.sparse import csr_matrix
from functools import partial
import pandas as pd
import numpy as np

class RestaurantEncoding(ClusterExplanationBase):

    def __init__(self, config):
        ClusterExplanationBase.__init__(self, config=config)

    def __khot__(self, x, max_usr_id):
        khv = np.zeros(max_usr_id, dtype=int)
        khv[x] = 1
        return khv

    def get_data(self, load=["RST_DATA", "RST_ENCODING", "N_USR_PCTG"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data:
            return dict_data

        # Si no existe, crear
        else:
            # Cargar los ficheros correspondientes
            ALL, _ = self.__basic_filtering__()

            # Obtener id de usuario ordenado por actividad (más reviews)
            usr_list = ALL.groupby("userId").like.count().sort_values(ascending=False).reset_index().rename(columns={"like": "new_id_user"})
            usr_list["new_id_user"] = list(range(len(usr_list)))
            ALL = ALL.merge(usr_list).drop(columns=["userId"]).rename(columns={"new_id_user": "userId"})

            # Obtener, para cada restaurante, los usuarios del x%
            max_usr_id = int(len(ALL.userId.unique())*self.CONFIG["pctg_usrs"])

            # Crear datos de restaurantes
            RST_DATA = ALL.groupby("restaurantId").apply(lambda x: pd.Series({"rst_name":x.rest_name.unique()[0], "num_images": x.num_images.sum(), "users":x.userId.values[x.userId.values<max_usr_id]})).reset_index()
            RST_DATA["n_usrs"] = RST_DATA["users"].apply(len)

            # Eliminar restaurantes con menos de X imagenes
            RST_DATA = RST_DATA.loc[RST_DATA.num_images>=self.CONFIG["min_imgs_per_rest"]]

            # Eliminar restaurantes con menos de X usuarios
            RST_DATA = RST_DATA.loc[RST_DATA.n_usrs>=self.CONFIG["min_usrs_per_rest"]]

            # Crear sparse con codificación
            RST_USRS = np.row_stack(RST_DATA.users.apply(partial(self.__khot__, max_usr_id=max_usr_id)).values)
            RST_USRS = csr_matrix(RST_USRS, dtype=np.int8)

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "RST_DATA", RST_DATA)
            to_pickle(self.DATASET_PATH, "RST_ENCODING", RST_USRS)
            to_pickle(self.DATASET_PATH, "N_USR_PCTG", max_usr_id)

            return self.get_dict_data(self.DATASET_PATH, load)
