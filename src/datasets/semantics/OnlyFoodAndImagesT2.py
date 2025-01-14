from src.datasets.semantics.DatasetSemantica import *
from src.Common import to_pickle, get_pickle, print_g, print_e

import os
import shutil

import numpy as np
import pandas as pd

class OnlyFoodAndImagesT2(DatasetSemantica):

    def __init__(self, config):
        DatasetSemantica.__init__(self, config=config)

    def __get_filtered_data__(self,save_path,items=["TRAIN", "TEST", "TEST2", "IMG", "IMG_VEC", "IMG_TEST", "USR_TMP", "REST_TMP"], verbose=True):

        def dropMultipleVisits(data):
            # Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
            multiple = data.groupby(["userId", "restaurantId"])["reviewId"].max().reset_index(name="last_reviewId")
            return data.loc[data.reviewId.isin(multiple.last_reviewId.values)].reset_index(drop=True)

        def __filter_and_save__(city):

            is_main_city = (c==self.CONFIG["city"])

            city_path = self.CONFIG["base_data_path"] + str(city) +"_data/"
            dataset_path = city_path + self.__class__.__name__ + "/"

            temp_file_name = "TRAIN"

            if not os.path.exists(dataset_path+temp_file_name):

                copy_from_path = "/media/HDD/pperez/TripAdvisor/"+city+"_data/OnlyFoodAndImages/"

                if os.path.exists(copy_from_path):

                    os.makedirs(dataset_path, exist_ok=True)
                    copy_files = ["TRAIN", "TEST", "IMG", "IMG_VEC", "USR_TMP", "REST_TMP"]

                    for fl in copy_files: shutil.copyfile(copy_from_path+fl, dataset_path+fl)

                else:
                    print_e("PATH DOES NOT EXIST")

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            TRAIN = get_pickle(dataset_path, "TRAIN")
            TEST = get_pickle(dataset_path, "TEST")
            RVW  = TRAIN.append(TEST)

            REST_TMP = get_pickle(dataset_path, "REST_TMP")
            IMG = get_pickle(dataset_path, "IMG")
            IMG_VEC = get_pickle(dataset_path, "IMG_VEC")

            return RVW, IMG, REST_TMP,IMG_VEC


        ################################################################################################################

        DICT = {}

        for i in items:
            if os.path.exists(save_path + i):DICT[i] = get_pickle(save_path, i)

        if len(DICT)!= len(items):

            data_dict = {}

            for c in tqdm(self.CONFIG["cities"]):
                data_dict[c] = __filter_and_save__(c)

            # Obtener la intersección de usuarios entre la ciudad actual y el resto
            # ---------------------------------------------------------------------------------------------------------------
            RVW = data_dict[self.CONFIG["city"]][0]
            REST_TMP = data_dict[self.CONFIG["city"]][2]

            RVWS_TEST = pd.DataFrame()
            IMGS_TEST = pd.DataFrame()

            main_usrs = set(RVW.userId.unique())
            intr_usrs = []

            oth_cities = self.CONFIG["cities"].copy()
            oth_cities.remove(self.CONFIG["city"])

            for c2 in oth_cities:
                c2_usrs = set(data_dict[c2][0].userId.unique())
                othr_usrs = list(c2_usrs.intersection(main_usrs))
                intr_usrs.extend(othr_usrs)

                c2_data_test = data_dict[c2][0].loc[data_dict[c2][0].userId.isin(othr_usrs)].copy()
                c2_data_test["city"] = c2
                c2_data_test["restaurantId"] = None
                RVWS_TEST = RVWS_TEST.append(c2_data_test)

                c2_imgs_test = data_dict[c2][1].loc[data_dict[c2][1].reviewId.isin(c2_data_test.reviewId)].copy()
                c2_imgs_test["city"] = c2
                c2_imgs_test["vector"] = data_dict[c2][3][c2_imgs_test.id_img.values,:].tolist()

                IMGS_TEST = IMGS_TEST.append(c2_imgs_test)

            intr_usrs = list(set(intr_usrs))

            #Añadir a test los usuarios de la ciudad actual (no hace falta eliminarlos de train)
            main_city_test = RVW.loc[RVW.userId.isin(intr_usrs)].copy()
            main_city_test["city"] = self.CONFIG["city"]
            RVWS_TEST  = RVWS_TEST.append(main_city_test)

            RVWS_TEST = RVWS_TEST.sample(frac=1).reset_index(drop=True)

            # Obtener ID para ONE-HOT de usuarios y restaurantes
            # --------------------------------------------------------------------------------------------------------------

            RVWS_TEST = RVWS_TEST.merge(REST_TMP, left_on="restaurantId",right_on='real_id', how='left')
            RVWS_TEST = RVWS_TEST[['date', 'images', 'restaurantId', 'reviewId','url', 'userId', 'num_images', 'rest_name', 'city', 'id_restaurant_y']]
            RVWS_TEST = RVWS_TEST.rename(columns={"id_restaurant_y": "id_restaurant"})
            to_pickle(save_path, "TEST2", RVWS_TEST)
            to_pickle(save_path, "IMG_TEST", IMGS_TEST)

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

    def get_data(self, load=["IMG", "IMG_VEC", "IMG_TEST", "N_USR", "V_IMG", "TRAIN", "TRAIN_RST_IMG", "RST_ADY", "TEST", "TEST2"]):

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

        if os.path.exists(file_path) and len(os.listdir(file_path)) == 13:

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

    def test_baseline_hdp(self):

        def get_positions(rest_popularity, relevant, n_relevant):

            rlvnt_pos = {}

            for rlv in relevant:
                tmp = np.argwhere(rest_popularity == rlv).flatten()[0]
                rlvnt_pos[rlv] = tmp

            first_pos = np.min(list(rlvnt_pos.values()))

            return rest_popularity[:n_relevant], 0, [], first_pos

        n_relevant = 1

        # Cargar los datos de los usuarios apartados al principio (FINAL_USRS)
        FINAL_USRS = self.DATA["TEST2"]
        #FINAL_USRS = FINAL_USRS.merge(self.DATA["IMG"], on="reviewId")

        # Restaurantes por popularidad
        rest_popularity = self.DATA["TRAIN"].id_restaurant.value_counts().reset_index().rename(
            columns={"index": "id_restaurant", "id_restaurant": "n_reviews"}).id_restaurant.values

        #################################################################################

        ret = []
        rest_rec = []
        rest_rel = []

        print("=" * 100)
        print("\n")

        for id, r in tqdm(FINAL_USRS.groupby("userId")):
            uid = r["userId"].values[0]
            relevant = r["id_restaurant_y"].unique()

            n_revs = len(r.reviewId.unique())
            n_imgs = len(r)

            img_idxs = r.id_img.to_list()

            retrieved, n_m, imgs, first_pos = get_positions(rest_popularity, relevant, n_relevant)

            acierto = int(first_pos < n_relevant)

            rest_relevant = r.loc[r.id_restaurant_y.isin(relevant)].rest_name.unique().tolist()
            rest_retrieved = self.DATA["TRAIN"].loc[self.DATA["TRAIN"].id_restaurant.isin(retrieved)].rest_name.unique().tolist()
            img_relevant = img_idxs
            img_retrieved = list(imgs)

            rest_rec.extend(retrieved)
            rest_rel.extend(relevant)

            intersect = len(set(retrieved).intersection(set(relevant)))

            prec = intersect / len(retrieved)
            rec = intersect / len(relevant)

            f1 = 0
            if (prec > 0 or rec > 0):
                f1 = 2 * ((prec * rec) / (prec + rec))

            ret.append((uid, first_pos, acierto, n_revs, n_imgs, prec, rec, f1, n_m, rest_relevant, rest_retrieved,
                        img_relevant, img_retrieved))

        ret = pd.DataFrame(ret,
                           columns=["user", "first_pos", "acierto", "n_revs", "n_imgs", "precision", "recall", "F1",
                                    "#recov", "rest_relevant", "rest_retrieved", "img_relevant", "img_retrieved"])

        pr = ret["precision"].mean()
        rc = ret["recall"].mean()
        f1 = ret["F1"].mean()

        print("\n")
        print(("%f\t%f\t%f\t%f\t%f") % (pr, rc, f1, ret["#recov"].mean(), ret["#recov"].std()))
        print(("%d\t%f") % (ret.acierto.sum(), ret.acierto.sum() / ret.acierto.count()))
        print(("%f\t%f\t%f") % (ret.first_pos.mean(), ret.first_pos.median(), ret.first_pos.std()))

        # Desglosar resultados por número de restaurantes

        ret["n_rest"] = ret.rest_relevant.apply(lambda x: len(x))

        desglose = []

        for n_r, rdata in ret.groupby("n_rest"):
            desglose.append((n_r, len(rdata), rdata["first_pos"].median(), rdata["F1"].mean(), rdata["acierto"].sum()))
            # print("%d\t%d\t%f\t%f\t%f" % (n_r, len(rdata),rdata["first_pos"].median(), rdata["F1"].mean(), rdata["acierto"].sum()))

        desglose = pd.DataFrame(desglose, columns=["n_rsts", "n_casos", "median", "f1", "aciertos"])

        desglose["n_casos_sum"] = (desglose["n_casos"].sum() - desglose["n_casos"].cumsum()).shift(1, fill_value=desglose["n_casos"].sum())
        desglose["aciertos_sum"] = (desglose["aciertos"].sum() - desglose["aciertos"].cumsum()).shift(1, fill_value=desglose["aciertos"].sum())
        desglose["prctg"] = desglose["aciertos_sum"] / desglose["n_casos_sum"]

        print("\n")

        for i in [1, 2, 3, 4]:
            tmp = desglose.loc[desglose.n_rsts == i][["n_casos_sum", "prctg"]]
            print("%d\t%d\n%f" % (i, tmp.n_casos_sum.values[0], tmp.prctg.values[0]))

        print("\n")

        # desglose.to_clipboard(excel=True)
