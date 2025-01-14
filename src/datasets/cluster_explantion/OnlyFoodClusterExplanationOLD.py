from src.datasets.DatasetClass import *
from src.Common import to_pickle, get_pickle, print_g

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class OnlyFoodClusterExplanation(DatasetClass):

    def __init__(self, config):
        DatasetClass.__init__(self, config=config)

    def __get_filtered_data__(self, save_path, items=["TRAIN", "TEST", "IMG", "IMG_VEC", "USR_TMP", "REST_TMP"], verbose=True):

        def dropMultipleVisits(data):
            # Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
            multiple = data.groupby(["userId", "restaurantId"])["reviewId"].max().reset_index(name="last_reviewId")
            return data.loc[data.reviewId.isin(multiple.last_reviewId.values)].reset_index(drop=True)

        ################################################################################################################

        DICT = {}

        for i in items:
            if os.path.exists(save_path + i):DICT[i] = get_pickle(save_path, i)

        if len(DICT) != len(items):

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

            # ---------------------------------------------------------------------------------------------------------------
            # Quedarse con las reviews positivas (>30)
            # --------------------------------------------------------------------------------------------------------------

            if self.CONFIG["only_positives"]:
                RVW = RVW.loc[RVW["like"] == 1]
                IMG = IMG.loc[IMG.review.isin(RVW.reviewId.unique())]

            # ---------------------------------------------------------------------------------------------------------------
            # Añadir URL a imágenes
            # --------------------------------------------------------------------------------------------------------------

            IMG = IMG.merge(RVW[["reviewId", "restaurantId", "images"]], left_on="review", right_on="reviewId", how="left")
            IMG["url"] = IMG.apply(lambda x: x.images[x.image]['image_url_lowres'], axis=1)
            IMG = IMG[["reviewId","restaurantId","image","url","vector","comida"]]

            # ---------------------------------------------------------------------------------------------------------------
            # ELIMINAR REVIEWS QUE NO TENGAN FOTO
            # --------------------------------------------------------------------------------------------------------------

            # RVW = RVW.loc[RVW.num_images>0]

            # ---------------------------------------------------------------------------------------------------------------
            # Quedarse con ultima review de los usuarios en caso de tener valoraciones diferentes (mismo rest)
            # --------------------------------------------------------------------------------------------------------------

            RVW = dropMultipleVisits(RVW)
            IMG = IMG.loc[IMG.reviewId.isin(RVW.reviewId)]

            # ---------------------------------------------------------------------------------------------------------------
            # Eliminar fotos que no sean de comida
            # --------------------------------------------------------------------------------------------------------------

            IMG_NO = IMG.loc[IMG.comida==0]
            IMG = IMG.loc[IMG.comida==1].reset_index(drop=True)

            # Eliminar las reviews que tienen todas sus fotos de "NO COMIDA"
            IMG_NO_NUM = IMG_NO.groupby("reviewId").image.count().reset_index(name="drop")
            IMG_NO_NUM = IMG_NO_NUM.merge(RVW.loc[RVW.reviewId.isin(IMG_NO_NUM.reviewId.values)][["reviewId", "num_images"]], on="reviewId")
            IMG_NO_NUM["delete"] = IMG_NO_NUM["drop"] == IMG_NO_NUM["num_images"]

            RVW = RVW.loc[~RVW.reviewId.isin(IMG_NO_NUM.loc[IMG_NO_NUM.delete == True].reviewId.values)]

            # En las otras se actualiza el número de fotos
            for _, r in IMG_NO_NUM.loc[IMG_NO_NUM.delete==False].iterrows():
                RVW.loc[RVW.reviewId == r.reviewId, "num_images"] = RVW.loc[RVW.reviewId == r.reviewId, "num_images"]-r["drop"]

            # ---------------------------------------------------------------------------------------------------------------
            # Separar en TRAIN y TEST
            # ---------------------------------------------------------------------------------------------------------------

            ALL_USRS = RVW.userId.unique()
            np.random.seed(self.CONFIG["seed"])
            np.random.shuffle(ALL_USRS)

            USRS_TRAIN = ALL_USRS[:len(ALL_USRS)//2]
            USRS_TEST = ALL_USRS[len(ALL_USRS)//2:]

            RVWS_TRAIN = RVW.loc[RVW.userId.isin(USRS_TRAIN)]
            RSTS_TRAIN = set(RVWS_TRAIN.restaurantId.unique())

            RVWS_TEST = RVW.loc[RVW.userId.isin(USRS_TEST)]
            RSTS_TEST = set(RVWS_TEST.restaurantId.unique())

            '''
            MOVE_TO_TRAIN = list(RSTS_TEST-RSTS_TRAIN)
            RVWS_TRAIN = RVWS_TRAIN.append(RVWS_TEST.loc[RVWS_TEST.restaurantId.isin(MOVE_TO_TRAIN)])
            RVWS_TEST = RVWS_TEST.loc[~RVWS_TEST.restaurantId.isin(MOVE_TO_TRAIN)]
            '''

            MOVE_TO_TRAIN = list(RSTS_TEST-RSTS_TRAIN)
            MOVE_TO_TRAIN_USRS = RVWS_TEST.loc[RVWS_TEST.restaurantId.isin(MOVE_TO_TRAIN)].userId.values
            RVWS_TRAIN = RVWS_TRAIN.append(RVWS_TEST.loc[RVWS_TEST.userId.isin(MOVE_TO_TRAIN_USRS)])
            RVWS_TEST = RVWS_TEST.loc[~RVWS_TEST.userId.isin(MOVE_TO_TRAIN_USRS)]

            # ---------------------------------------------------------------------------------------------------------------
            # Stats de los conjuntos
            # --------------------------------------------------------------------------------------------------------------

            '''
            print("ONLY POSITIVES:", self.CONFIG["only_positives"])
            def get_stats(DS, NAME):
                print(NAME)
                print("\t · #REVIEWS:",len(DS))
                print("\t · #RESTAURANTS:", len(DS.rest_name.unique()))
                print("\t · AVG RVW X RST:", DS.groupby("rest_name").id.count().mean())
                print("\t · AVG IMG X RST:", DS.groupby("rest_name").num_images.sum().mean())
                print("\t · #USRS:", len(DS.userId.unique()))
                print("\t · AVG RVW X USR:", DS.groupby("userId").id.count().mean())
                print("\t · AVG IMG X USR:", DS.groupby("userId").num_images.sum().mean())
                print("\t · #IMGS:", DS.num_images.sum())

            get_stats(RVWS_TRAIN, "TRAIN")
            get_stats(RVWS_TEST, "TEST")
            '''

            # ---------------------------------------------------------------------------------------------------------------
            # Obtener ID para ONE-HOT de usuarios y restaurantes
            # --------------------------------------------------------------------------------------------------------------

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
            IMG = IMG.drop(columns=["restaurantId", "real_id"])

            # Añadir columna identificar imágenes de test
            IMG["test"] = False
            IMG.loc[IMG.reviewId.isin(RVWS_TEST.reviewId), "test"] = True

            RVWS_TRAIN = RVWS_TRAIN[['date', 'images', 'language', 'rating', 'restaurantId', 'reviewId', 'text', 'title', 'url', 'userId', 'num_images', 'id_user', 'id_restaurant','rest_name', 'like']]

            # ---------------------------------------------------------------------------------------------------------------
            # Separar vectores de imágenes
            # --------------------------------------------------------------------------------------------------------------

            IMG = IMG.reset_index(drop=True)
            IMG["id_img"] = range(len(IMG))
            IMG_VEC = np.row_stack(IMG.vector.values)
            IMG = IMG.drop(columns=['vector'])

            to_pickle(save_path, "TEST", RVWS_TEST)
            to_pickle(save_path, "TRAIN", RVWS_TRAIN)
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

    def get_paper_stats(self):

        def dropMultipleVisits(data):
            # Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
            multiple = data.groupby(["userId", "restaurantId"])["reviewId"].max().reset_index(name="last_reviewId")
            return data.loc[data.reviewId.isin(multiple.last_reviewId.values)].reset_index(drop=True)

        ################################################################################################################

        IMG = pd.read_pickle(self.CONFIG["data_path"] + "img-hd-densenet.pkl")
        RVW = pd.read_pickle(self.CONFIG["data_path"] + "reviews.pkl")
        RST = pd.read_pickle(self.CONFIG["data_path"] + "restaurants.pkl")
        RST.rename(columns={"name": "rest_name"}, inplace=True)

        if "index" in RVW.columns: RVW = RVW.drop(columns="index")

        IMG['review'] = IMG.review.astype(int)
        RST["id"] = RST.id.astype(int)

        RVW["reviewId"] = RVW.reviewId.astype(int)
        RVW["restaurantId"] = RVW.restaurantId.astype(int)

        RVW = RVW.merge(RST[["id", "rest_name"]], left_on="restaurantId", right_on="id", how="left")

        RVW["num_images"] = RVW.images.apply(lambda x: len(x))
        RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
        RVW = RVW.loc[(RVW.userId != "")]

        # ---------------------------------------------------------------------------------------------------------------
        # Quedarse con las reviews positivas (>30)
        # --------------------------------------------------------------------------------------------------------------

        if self.CONFIG["only_positives"]:
            RVW = RVW.loc[RVW["like"] == 1]
            IMG = IMG.loc[IMG.review.isin(RVW.reviewId.unique())]

        # ---------------------------------------------------------------------------------------------------------------
        # Añadir URL a imágenes
        # --------------------------------------------------------------------------------------------------------------

        IMG = IMG.merge(RVW[["reviewId", "restaurantId", "images"]], left_on="review", right_on="reviewId", how="left")
        IMG["url"] = IMG.apply(lambda x: x.images[x.image]['image_url_lowres'], axis=1)
        IMG = IMG[["reviewId", "restaurantId", "image", "url", "vector", "comida"]]

        # ---------------------------------------------------------------------------------------------------------------
        # ELIMINAR REVIEWS QUE NO TENGAN FOTO
        # --------------------------------------------------------------------------------------------------------------

        # RVW = RVW.loc[RVW.num_images>0]

        # ---------------------------------------------------------------------------------------------------------------
        # Quedarse con ultima review de los usuarios en caso de tener valoraciones diferentes (mismo rest)
        # --------------------------------------------------------------------------------------------------------------

        RVW = dropMultipleVisits(RVW)
        IMG = IMG.loc[IMG.reviewId.isin(RVW.reviewId)]

        # ---------------------------------------------------------------------------------------------------------------
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
            RVW.loc[RVW.reviewId == r.reviewId, "num_images"] = RVW.loc[RVW.reviewId == r.reviewId, "num_images"] - r["drop"]

        n_usrs = len(RVW.userId.unique())
        n_rst = len(RVW.restaurantId.unique())
        n_rvws = len(RVW)
        n_photos = RVW.num_images.sum()

        print(self.CONFIG["city"])
        print("\t".join(list(map(str, [n_usrs, n_rst, n_rvws, n_photos]))))

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

        file_path = self.DATASET_PATH

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

    def get_stats(self):

        train_data = self.DATA["TRAIN"].copy()

        # Eliminar restaurantes con menos de 5 imágenes
        rst_n_img = train_data.groupby("restaurantId").apply(lambda x: pd.Series({"num_images": x.num_images.sum()})).reset_index()
        train_data = train_data.loc[train_data.restaurantId.isin(rst_n_img.loc[rst_n_img.num_images >= 5].restaurantId)]

        usr_list = train_data.groupby("id_user").like.count().sort_values(ascending=False).reset_index().rename(columns={"like": "new_id_user"})
        usr_list["new_id_user"] = list(range(len(usr_list)))
        train_data = train_data.merge(usr_list).drop(columns=["id_user"]).rename(columns={"new_id_user": "id_user"})

        n_usrs = len(train_data.id_user.unique())

        base_savepath = 'out/cluster_explanation/stats/%s/' % (self.CONFIG["city"])
        os.makedirs(base_savepath, exist_ok=True)

        res = []

        # Ver como quedan los conjuntos al variar el porcentaje de usuarios
        for p in np.asarray(range(35,0,-5))/100:
            n_usrs_pctg = int(n_usrs*p)

            # Eliminar los usuarios fuera del X por ciento
            train_data_pctg = train_data.loc[train_data.id_user<n_usrs_pctg]
            assert n_usrs_pctg == len(train_data_pctg.userId.unique())

            # Eliminar los restaurantes con menos de "nu" usuarios
            rst_n_usr = train_data_pctg.groupby("id_restaurant").apply(lambda x: pd.Series({"num_usrs": len(x.id_user.unique())})).reset_index()
            for nu in range(10,0,-1):
                train_data_pctg_nu = train_data_pctg.loc[train_data_pctg.id_restaurant.isin(rst_n_usr.loc[rst_n_usr.num_usrs >= nu].id_restaurant)]

                # Se calculan las stats
                stats_by_rst = train_data_pctg_nu.groupby("id_restaurant").apply(lambda x: pd.Series({'num_images': x.num_images.sum(), "num_usrs": len(x.id_user.unique())})).reset_index()

                # Plots de imgs per rest y usrs per rest
                img_pctg_plt = sns.distplot(stats_by_rst.num_images, kde=False, bins=range(100))
                img_pctg_plt.set_title("Imágenes por restaurante [ %d%% | min_usrs_pr_rest: %d ]" % (int(p*100),nu))
                img_pctg_plt.set_ylabel("Número de restaurantes")
                img_pctg_plt.set_xlabel("Número de imágenes")
                img_pctg_plt.set_xticks(range(0, 100, 5))
                img_pctg_plt.grid()
                plt.savefig(base_savepath+"imgs_per_rest_%d_%d.jpg" % (int(p*100),nu))
                plt.clf()

                usr_pctg_plt = sns.distplot(stats_by_rst.num_usrs, kde=False, bins=range(100))
                usr_pctg_plt.set_title("Usuarios por restaurante [ %d%% | min_usrs_pr_rest: %d ]" % (int(p*100),nu))
                usr_pctg_plt.set_ylabel("Número de restaurantes")
                usr_pctg_plt.set_xlabel("Número de usuarios")
                usr_pctg_plt.set_xticks(range(0, 100, 5))
                usr_pctg_plt.grid()
                plt.savefig(base_savepath+"usrs_per_rest_%d_%d.jpg" % (int(p*100),nu))
                plt.clf()

                res.append((int(p*100),nu, n_usrs_pctg,train_data_pctg_nu.id_restaurant.value_counts().size, train_data_pctg_nu.num_images.sum()))

        res = pd.DataFrame(res, columns=["pctg_usrs","min_usrs_per_rst","n_usrs","n_rsts","n_imgs"])
        res.to_excel(base_savepath+"%s_stats.xlsx"%self.CONFIG["city"])