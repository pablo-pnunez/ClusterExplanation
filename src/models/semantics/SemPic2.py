# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd

from src.models.semantics.SemPic import *
from scipy.spatial.distance import cdist

########################################################################################################################

class SemPic2(SemPic):

    def __init__(self,config, dataset):
        SemPic.__init__(self, config=config, dataset=dataset)

    def __get_md5__(self):
        # Retorna el MD5
        return hashlib.md5(str(self.CONFIG["model"]).encode()).hexdigest(), self.CONFIG["model"]

    def get_model(self):

        self.DATASET.DATA["N_USR"]= int( self.DATASET.DATA["N_USR"] * self.CONFIG["model"]["pctg_usrs"])
        print_w("Updating user number to %d" % self.DATASET.DATA["N_USR"])

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = BatchNormalization()(input)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(128, name="img_emb")(x)
        x = BatchNormalization()(x)
        output = Dense(self.DATASET.DATA["N_USR"], activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[self.precision, self.recall, self.f1, "accuracy"])

        return model

    def find_image_semantics(self):

        all_img_embs = self.__get_image_encoding__(encoding="emb") # Obtener embeddings de todas las imágenes

        carne_normal = self.DATASET.DATA["IMG"].loc[(self.DATASET.DATA["IMG"].reviewId==194249123) & (self.DATASET.DATA["IMG"].image==1)].id_img.values[0]
        carne_normal = all_img_embs[carne_normal,:]
        carne_fina   = self.DATASET.DATA["IMG"].loc[(self.DATASET.DATA["IMG"].reviewId==251743984) & (self.DATASET.DATA["IMG"].image==2)].id_img.values[0]
        carne_fina   = all_img_embs[carne_fina,:]

        pescado_normal = self.DATASET.DATA["IMG"].loc[(self.DATASET.DATA["IMG"].reviewId==122143281) & (self.DATASET.DATA["IMG"].image==1)].id_img.values[0]
        pescado_normal = all_img_embs[pescado_normal,:]

        vc = carne_normal +(carne_normal)
        dstncs = cdist([vc], all_img_embs)
        min_idx = np.argmin(dstncs)
        min_img = self.DATASET.DATA["IMG"].loc[self.DATASET.DATA["IMG"].id_img == min_idx]

        print(min_img)

    ####################################################################################################################

    class Sequence(Sequence):

        def __init__(self, model):

            self.MODEL = model
            self.N_RESTAURANTS = len(self.MODEL.DATASET.DATA["RST_ADY"])
            self.BATCH_SIZE = self.MODEL.CONFIG["model"]["batch_size"]

            self.init_data()

        def init_data(self):

            train_data = self.MODEL.DATASET.DATA['TRAIN'].copy()
            n_usrs = self.MODEL.DATASET.DATA["N_USR"]

            if "active_usrs" in self.MODEL.CONFIG["model"].keys() and self.MODEL.CONFIG["model"]["active_usrs"] is True:
                usr_list = self.MODEL.DATASET.DATA['TRAIN'].groupby("id_user").like.count().sort_values(ascending=False).reset_index().rename(columns={"like":"new_id_user"})
                usr_list["new_id_user"] = list(range(len(usr_list)))
                train_data = train_data.merge(usr_list).drop(columns=["id_user"]).rename(columns={"new_id_user":"id_user"})

            self.SEQUENCE_DATA = train_data
            self.KHOT = MultiLabelBinarizer(classes=list(range(self.MODEL.DATASET.DATA["N_USR"])))

            x = []
            y = []
            r = []

            for id_r, rows in tqdm(train_data.groupby("id_restaurant"), desc="USRS DATA"):
                rst_imgs = self.MODEL.DATASET.DATA["TRAIN_RST_IMG"].loc[id_r]

                # Obtener usuarios
                rltd_usrs = rows.id_user.unique()
                rltd_usrs = rltd_usrs[np.argwhere(rltd_usrs < n_usrs).flatten()]

                r.extend([id_r]* len(rst_imgs))
                x.extend(rst_imgs)
                y.extend([list(rltd_usrs)] * len(rst_imgs))

            ############################################################################################################

            ret = pd.DataFrame(list(zip(r,x, y)), columns=["id_restaurant","id_img", "output"]).sample(frac=1)

            ############################################################################################################

            self.ALL_DATA = ret

            if (len(ret) > self.BATCH_SIZE):
                self.BATCH = np.array_split(ret, len(ret) // self.BATCH_SIZE)

            else:
                self.BATCH = np.array_split(ret, 1)

        def __len__(self):
            return len(self.BATCH)

        def __getitem__(self, idx):

            btch = self.BATCH[idx]

            x = self.MODEL.DATASET.DATA['IMG_VEC'][btch.id_img.values]
            y = self.KHOT.fit_transform(btch.output.values)

            return (x, y)

    ####################################################################################################################

    def __get_model__(self, layer_name ="img_emb"):

        self.MODEL.load_weights(self.MODEL_PATH + "weights")
        sub_model = Model(inputs=[self.MODEL.get_layer("in").input], outputs=[self.MODEL.get_layer(layer_name).output])

        return sub_model

    def test2(self, encoding="",  n_relevant=1, previous_result=None, log2file=False, baseline=False):

        def getRelevantRestaurants(number, current, others, others_data, relevant, popularity=None):

            if popularity is None:

                if "dense" in encoding:
                    dists = cdist([current], others)[0]
                    arg_dist_sort = np.argsort(dists).flatten()

                elif "emb" in encoding:
                    dists = [np.dot(current, others[i, :]) for i in range(len(others))]
                    arg_dist_sort = np.argsort((np.asarray(dists) * -1))
                else:
                    return Exception

                all_rsts_ordered = list(dict.fromkeys(np.asarray(others_data)[arg_dist_sort]))

                n_fts = number  # Número de fotos más cercanas
                ret = []

                while len(ret) < number:
                    idxs = arg_dist_sort[:n_fts]
                    ret = list(dict.fromkeys(np.asarray(others_data)[idxs]))  # Eliminar duplicados conservando el orden
                    n_fts += 1

            else:
                all_rsts_ordered = popularity
                ret = popularity[:number]
                n_fts = 0
                idxs = []

            rlvnt_pos = {}

            for rlv in relevant:
                tmp = np.argwhere(all_rsts_ordered == rlv).flatten()[0]
                rlvnt_pos[rlv] = tmp

            first_pos = np.min(list(rlvnt_pos.values()))

            return ret, n_fts - 1, idxs, first_pos


        # Almacenar la salida en fichero
        log_file_path = self.MODEL_PATH + "test2.txt"
        if log2file and os.path.exists(log_file_path): print_e("Test already done"); exit(0)
        if log2file: sys.stdout = open(log_file_path, "w")

        # Modelo entrenado

        model = self.__get_model__()

        # Cargar los datos de los usuarios apartados al principio (FINAL_USRS)
        FINAL_USRS = self.DATASET.DATA["TEST2"]

        TRAIN_IMGS = self.DATASET.DATA["IMG"].loc[ self.DATASET.DATA["IMG"].test==False]
        TRAIN_DNSN = self.DATASET.DATA["IMG_VEC"][TRAIN_IMGS.id_img.values,:]

        train_img_rest = TRAIN_IMGS.id_restaurant.to_list()

        #OBTENER EMBBEDINGS
        TEST_2_IMGS = self.DATASET.DATA["IMG_TEST"].reset_index(drop=True)
        TEST_2_DNSN = np.row_stack(TEST_2_IMGS.vector.values)

        if encoding is "emb":
            TRAIN_EMBS = model.predict(TRAIN_DNSN, batch_size=self.CONFIG["model"]["batch_size"])
            TEST_2_EMBS = model.predict(TEST_2_DNSN, batch_size=self.CONFIG["model"]["batch_size"])
        if encoding is "dense":
            TRAIN_EMBS = TRAIN_DNSN
            TEST_2_EMBS = TEST_2_DNSN

        if baseline:
            rest_popularity = self.DATASET.DATA["TRAIN"].id_restaurant.value_counts().reset_index().rename(columns={"index": "id_restaurant", "id_restaurant": "n_reviews"}).id_restaurant.values

        #################################################################################

        ret = []
        rest_rec = []
        rest_rel = []

        for id, r in tqdm(FINAL_USRS.groupby("userId")):

            uid = r["userId"].values[0]
            relevant = r["id_restaurant"].dropna().unique()

            rvw_idxs = r.loc[r.city != self.DATASET.CONFIG["city"]].reviewId.to_list()
            rvw_imgs_idx = TEST_2_IMGS.loc[TEST_2_IMGS.reviewId.isin(rvw_idxs)].index.values
            rvw_embs = TEST_2_EMBS[rvw_imgs_idx,:]

            n_revs = len(relevant)
            n_imgs = len(rvw_imgs_idx)

            mean_img = np.mean(rvw_embs, axis=0) #Centroide

            if baseline: retrieved, n_m, imgs, first_pos = getRelevantRestaurants(n_relevant, mean_img, TRAIN_EMBS, train_img_rest, relevant,rest_popularity)
            else: retrieved, n_m, imgs, first_pos = getRelevantRestaurants(n_relevant, mean_img, TRAIN_EMBS, train_img_rest, relevant)

            acierto = int(first_pos < n_relevant)

            rest_relevant = r.loc[r.id_restaurant.isin(relevant)].rest_name.unique().tolist()
            rest_retrieved = self.DATASET.DATA["TRAIN"].loc[self.DATASET.DATA["TRAIN"].id_restaurant.isin(retrieved)].rest_name.unique().tolist()
            img_relevant = rvw_imgs_idx
            img_retrieved = list(imgs)

            rest_rec.extend(retrieved)
            rest_rel.extend(relevant)

            intersect = len(set(retrieved).intersection(set(relevant)))

            prec = intersect / len(retrieved)
            rec = intersect / len(relevant)

            f1 = 0
            if (prec > 0 or rec > 0):
                f1 = 2 * ((prec * rec) / (prec + rec))

            ret.append((uid, first_pos, acierto, n_revs, n_imgs, prec, rec, f1, n_m, rest_relevant, rest_retrieved, img_relevant, img_retrieved))

        ret = pd.DataFrame(ret, columns=["user", "first_pos", "acierto", "n_revs", "n_imgs", "precision", "recall", "F1", "#recov", "rest_relevant", "rest_retrieved", "img_relevant", "img_retrieved"])
        ret.to_excel("docs/" + encoding.lower() + "_test.xlsx")

        pr = ret["precision"].mean()
        rc = ret["recall"].mean()
        f1 = ret["F1"].mean()

        print("%f\t%f\t%f\t%f\t%f" % (pr, rc, f1, ret["#recov"].mean(), ret["#recov"].std()))
        print("%d\t%f" % (ret.acierto.sum(), ret.acierto.sum() / ret.acierto.count()))
        print("%f\t%f\t%f" % (ret.first_pos.mean(), ret.first_pos.median(), ret.first_pos.std()))

        # Desglosar resultados por número de restaurantes

        ret["n_rest"] = ret.rest_relevant.apply(lambda x: len(x))

        desglose = []

        for n_r, rdata in ret.groupby("n_rest"):
            desglose.append((n_r, len(rdata), rdata["first_pos"].median(), rdata["F1"].mean(), rdata["acierto"].sum()))
            # print("%d\t%d\t%f\t%f\t%f" % (n_r, len(rdata),rdata["first_pos"].median(), rdata["F1"].mean(), rdata["acierto"].sum()))

        desglose = pd.DataFrame(desglose, columns=["n_rsts", "n_casos", "median", "f1", "aciertos"])

        desglose["n_casos_sum"] = (desglose["n_casos"].sum() - desglose["n_casos"].cumsum()).shift(1,
                                                                                                   fill_value=desglose[
                                                                                                       "n_casos"].sum())
        desglose["aciertos_sum"] = (desglose["aciertos"].sum() - desglose["aciertos"].cumsum()).shift(1, fill_value=
        desglose["aciertos"].sum())
        desglose["prctg"] = desglose["aciertos_sum"] / desglose["n_casos_sum"]

        print("·" * 100)

        for i in [1, 2, 3, 4]:
            if desglose.n_rsts.min() > i: continue
            tmp = desglose.loc[desglose.n_rsts == i][["n_casos_sum", "prctg"]]
            print("%d\t%d\n%f" % (i, tmp.n_casos_sum.values[0], tmp.prctg.values[0]))

        # desglose.to_clipboard(excel=True)

        if previous_result == None:
            previous_result = {encoding: ret}
        else:
            previous_result[encoding] = ret

        # Cerrar fichero de salida
        if log2file: sys.stdout.close()

        return previous_result