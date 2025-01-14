from src.datasets.semantics.DatasetSemantica import *
from src.Common import to_pickle, get_pickle, print_g

import os

import numpy as np
import pandas as pd

class OnlyFood(DatasetSemantica):

    def __init__(self, config):
        DatasetSemantica.__init__(self, config=config)

    def __get_filtered_data__(self,save_path,items=["TRAIN","TEST","IMG","IMG_VEC","USR_TMP","REST_TMP"],verbose=True):

        def dropMultipleVisits(data):
            # Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
            multiple = data.groupby(["userId", "restaurantId"])["reviewId"].max().reset_index(name="last_reviewId")
            return data.loc[data.reviewId.isin(multiple.last_reviewId.values)].reset_index(drop=True)

        def dropNoImgRests(data):

            DROP_RST = data.groupby("restaurantId")["num_images"].apply(lambda x: np.sum(x)==0).reset_index(name="drop")
            print("Se eliminan %d de %d restaurantes del conjunto" % (len(DROP_RST.loc[DROP_RST["drop"]==True]), len(DROP_RST)))

            data = data.loc[~data.restaurantId.isin(DROP_RST.loc[DROP_RST["drop"]==True].restaurantId.unique())]

            return data

        ################################################################################################################

        DICT = {}

        for i in items:
            if os.path.exists(save_path + i):DICT[i] = get_pickle(save_path, i)

        if len(DICT)!= len(items):

            IMG = pd.read_pickle(self.CONFIG["data_path"] + "img-hd-densenet.pkl")
            RVW = pd.read_pickle(self.CONFIG["data_path"] + "reviews.pkl")
            RST = pd.read_pickle(self.CONFIG["data_path"] + "restaurants.pkl")
            RST.rename(columns={"name": "rest_name"}, inplace=True)

            if "index" in RVW.columns:RVW = RVW.drop(columns="index")

            IMG['review'] = IMG.review.astype(int)
            RST["id"] = RST.id.astype(int)

            RVW["reviewId"] = RVW.reviewId.astype(int)
            RVW["restaurantId"] = RVW.restaurantId.astype(int)

            RVW = RVW.merge(RST[["id","rest_name"]], left_on="restaurantId", right_on="id", how="left")

            RVW["num_images"] = RVW.images.apply(lambda x: len(x))
            RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
            RVW = RVW.loc[(RVW.userId != "")]

            # Añadir URL a imágenes
            # --------------------------------------------------------------------------------------------------------------

            IMG = IMG.merge(RVW[["reviewId", "restaurantId", "images"]], left_on="review", right_on="reviewId", how="left")
            IMG["url"] = IMG.apply(lambda x: x.images[x.image]['image_url_lowres'], axis=1)
            IMG = IMG[["reviewId","restaurantId","image","url","vector","comida"]]

            # Quedarse con ultima review de los usuarios en caso de tener valoraciones diferentes (mismo rest)
            # --------------------------------------------------------------------------------------------------------------

            RVW = dropMultipleVisits(RVW)
            IMG = IMG.loc[IMG.reviewId.isin(RVW.reviewId)]

            # Eliminar fotos que no sean de comida
            # --------------------------------------------------------------------------------------------------------------

            IMG_NO = IMG.loc[IMG.comida == 0]
            IMG = IMG.loc[IMG.comida == 1].reset_index(drop=True)

            # Eliminar las reviews que tienen todas sus fotos de "NO COMIDA"
            IMG_NO_NUM = IMG_NO.groupby("reviewId").image.count().reset_index(name="drop")
            IMG_NO_NUM = IMG_NO_NUM.merge(RVW.loc[RVW.reviewId.isin(IMG_NO_NUM.reviewId.values)][["reviewId", "num_images"]], on="reviewId")
            IMG_NO_NUM["delete"] = IMG_NO_NUM["drop"] == IMG_NO_NUM["num_images"]

            RVW = RVW.loc[~RVW.reviewId.isin(IMG_NO_NUM.loc[IMG_NO_NUM.delete == True].reviewId.values)]

            # En las otras se actualiza el número de fotos
            for _, r in IMG_NO_NUM.loc[IMG_NO_NUM.delete == False].iterrows():
                RVW.loc[RVW.reviewId == r.reviewId, "num_images"] = RVW.loc[RVW.reviewId == r.reviewId, "num_images"] - \
                                                                    r["drop"]

            # Eliminar usuarios que solo tengan un restaurante
            # ---------------------------------------------------------------------------------------------------------------

            DROP_USR = RVW.groupby("userId")["restaurantId"].apply(lambda x: len(np.unique(x))).reset_index(name="n_rests")
            DROP_USR = DROP_USR.loc[DROP_USR.n_rests == 1].userId.unique()

            RVW = RVW.loc[~RVW.userId.isin(DROP_USR)]
            IMG = IMG.loc[IMG.reviewId.isin(RVW.reviewId)]

            # Eliminar restaurantes sin fotos (al hacer esto se incumple la restricción anterior)
            # ---------------------------------------------------------------------------------------------------------------

            RVW = dropNoImgRests(RVW)

            # Separar 10% usuarios (del total,no de los que cumple las condiciones) con, fotos en todas sus reviews
            # --------------------------------------------------------------------------------------------------------------

            PCTG_USRS = int(len(RVW.userId.unique()) * .1)  # Cuantos usuarios van para test
            USR_IMG_COUNT = RVW.groupby("userId")["num_images"].min().reset_index(name="photos")
            FINAL_TEST_USRS = USR_IMG_COUNT.loc[USR_IMG_COUNT.photos > 0]
            if (len(FINAL_TEST_USRS) < PCTG_USRS): print("No hay suficientes usuarios para llegar al 10%% (%d), se seleccionan menos (%d)." % (PCTG_USRS, len(FINAL_TEST_USRS)))
            FINAL_TEST_USRS = FINAL_TEST_USRS.sample(min(PCTG_USRS, len(FINAL_TEST_USRS))).userId.to_list()
            FINAL_TEST_USRS = RVW.loc[RVW.userId.isin(FINAL_TEST_USRS)]

            RVW = RVW.loc[~RVW.userId.isin(FINAL_TEST_USRS.userId)]

            # Eliminar restaurantes sin fotos
            # ---------------------------------------------------------------------------------------------------------------

            RVW = dropNoImgRests(RVW)

            # Obtener ID para ONE-HOT de usuarios y restaurantes
            # --------------------------------------------------------------------------------------------------------------

            USR_TMP = pd.DataFrame(columns=["real_id", "id_user"])
            REST_TMP = pd.DataFrame(columns=["real_id", "id_restaurant"])

            # Obtener tabla real_id -> id para usuarios
            USR_TMP.real_id = RVW.sort_values("userId").userId.unique()
            USR_TMP.id_user = range(0, len(USR_TMP))

            # Obtener tabla real_id -> id para restaurantes
            REST_TMP.real_id = RVW.sort_values("restaurantId").restaurantId.unique()
            REST_TMP.id_restaurant = range(0, len(REST_TMP))

            # Mezclar datos
            RET = RVW.merge(USR_TMP, left_on='userId', right_on='real_id', how='inner')
            RET = RET.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')
            FINAL_TEST_USRS = FINAL_TEST_USRS.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='left')

            IMG = IMG.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')
            IMG = IMG.drop(columns=["restaurantId", "real_id"])

            # Si hay restaurantes que desaparecen en el conjunto de test, eliminarlos

            DROP_FINAL_USRS = FINAL_TEST_USRS.groupby("userId").id_restaurant.apply(lambda x: any(np.isnan(x))).reset_index(name="drop")
            print("Se eliminan %d de %d usuarios de test" % (len(DROP_FINAL_USRS.loc[DROP_FINAL_USRS["drop"] == True]), len(DROP_FINAL_USRS)))

            DROP_FINAL_USRS = DROP_FINAL_USRS.loc[DROP_FINAL_USRS["drop"] == True].userId
            DROP_FINAL_RVWS = FINAL_TEST_USRS.loc[FINAL_TEST_USRS.userId.isin(DROP_FINAL_USRS)].reviewId.to_list()
            FINAL_TEST_USRS = FINAL_TEST_USRS.loc[~FINAL_TEST_USRS.userId.isin(DROP_FINAL_USRS)]

            # Eliminar tambien las imágenes de las reviews eliminadas
            IMG = IMG.loc[~IMG.reviewId.isin(DROP_FINAL_RVWS)]
            # Añadir
            IMG["test"] = False
            IMG.loc[IMG.reviewId.isin(FINAL_TEST_USRS.reviewId), "test"] = True

            RVW = RET[['date', 'images', 'language', 'rating', 'restaurantId', 'reviewId', 'text', 'title', 'url', 'userId', 'num_images', 'id_user', 'id_restaurant', 'rest_name', 'like']]

            # Separar vectores de imágenes
            # --------------------------------------------------------------------------------------------------------------

            IMG = IMG.reset_index(drop=True)
            IMG["id_img"] = range(len(IMG))
            IMG_VEC = np.row_stack(IMG.vector.values)
            IMG = IMG.drop(columns=['vector'])

            to_pickle(save_path, "TEST", FINAL_TEST_USRS)
            to_pickle(save_path, "TRAIN", RVW)
            to_pickle(save_path, "IMG", IMG)
            to_pickle(save_path, "IMG_VEC", IMG_VEC)
            to_pickle(save_path, "USR_TMP", USR_TMP)
            to_pickle(save_path, "REST_TMP", REST_TMP)

            for i in items:
                if os.path.exists(save_path + i):DICT[i] = get_pickle(save_path, i)

        if verbose:
            print_g("-"*50, title=False)
            print_g(" TRAIN Rev  number: " + str(len(DICT["TRAIN"])))
            print_g(" TRAIN User number: " + str(len(DICT["TRAIN"].userId.unique())))
            print_g(" TRAIN Rest number: " + str(len(DICT["TRAIN"].restaurantId.unique())))
            print_g("-"*50, title=False)
            print_g(" TEST  Rev  number: " + str(len(DICT["TEST"])))
            print_g(" TEST  User number: " + str(len(DICT["TEST"].userId.unique())))
            print_g(" TEST  Rest number: " + str(len(DICT["TEST"].restaurantId.unique())))
            print_g("-"*50, title=False)

        return DICT

    def get_data(self, load=["IMG", "IMG_VEC", "N_USR", "V_IMG", "TRAIN", "TRAIN_RST_IMG", "RST_ADY", "TEST"]):

        def createSets(dictionary):

            def generateTrainItems(img):

                # Obtener para cada restaurante una lista de sus imágenes (ordenado por id de restaurante)
                rst_img = img.loc[img.test==False].groupby("id_restaurant").id_img.apply(lambda x: np.asarray(np.unique(x), dtype=int)).reset_index(name="imgs", drop=True)
                # Obtener para cada usuario una lista de los restaurantes a los que fué
                # usr_rsts = data.groupby("id_user").id_restaurant.apply(lambda x: np.unique(x)).reset_index(drop=True, name="rsts")

                return rst_img

            def get_rest_ady(data):

                rsts = np.sort(data.id_restaurant.unique())

                ret = []

                for r in rsts:
                    rc = data.loc[data.id_restaurant == r]
                    rc_u = rc.id_user.unique()

                    ro = data.loc[data.id_user.isin(rc_u)].groupby("id_restaurant").id_user.count().reset_index(
                        name=r).set_index("id_restaurant")
                    ro = ro.drop(index=r)
                    ret.append((r, ro.index.to_list()))

                ret = pd.DataFrame(ret, columns=["id_restaurant", "ady"])

                return ret

            # ------------------------------------------------------------------

            IMG = dictionary["IMG"]
            TRAIN = dictionary["TRAIN"]

            if not os.path.exists(file_path + "RST_ADY"):

                # Obtener para cada restaurante, los adyacentes
                RST_ADY = get_rest_ady(TRAIN)
                TRAIN_RST_IMG = generateTrainItems( IMG)

                to_pickle(file_path, "TRAIN_RST_IMG", TRAIN_RST_IMG);
                to_pickle(file_path, "RST_ADY", RST_ADY);

                del TRAIN_RST_IMG, RST_ADY

            else:
                print_g("TRAIN set already created, omitting...")

        ################################################################################################################


        # Mirar si ya existen los datos
        # --------------------------------------------------------------------------------------------------------------

        file_path = self.CONFIG["data_path"] + self.__class__.__name__+"/"

        if os.path.exists(file_path) and len(os.listdir(file_path)) == 11:

            print_g("Loading previous generated data...")

            DICT = {}

            for d in load:
                if os.path.exists(file_path + d):
                    DICT[d] = get_pickle(file_path, d)

            return DICT

        os.makedirs(file_path, exist_ok=True)

        DICT = self.__get_filtered_data__(file_path)

        # Crear conjuntos de TRAIN/DEV/TEST y GUARDAR
        # --------------------------------------------------------------------------------------------------------------

        createSets(DICT)

        # Almacenar pickles
        # --------------------------------------------------------------------------------------------------------------

        to_pickle(file_path, "N_RST", len(DICT["REST_TMP"]))
        to_pickle(file_path, "N_USR", len(DICT["USR_TMP"]))
        to_pickle(file_path, "V_IMG", DICT["IMG_VEC"].shape[1])

        # Cargar datos creados previamente
        # --------------------------------------------------------------------------------------------------------------

        DICT = {}

        for d in load:
            if os.path.exists(file_path + d):
                DICT[d] = get_pickle(file_path, d)

        return DICT

