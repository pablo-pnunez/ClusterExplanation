from src.datasets.OldDatasetClass import *

import numpy as np
import pandas as pd

from tqdm import tqdm

class DatasetSemantica(OldDatasetClass):

    def __init__(self, config):
        OldDatasetClass.__init__(self, config=config)

    def get_data(self):
        # Retorna un diccionario con los datos
        raise NotImplementedError

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
        FINAL_USRS = self.DATA["TEST"]
        FINAL_USRS = FINAL_USRS.merge(self.DATA["IMG"], on="reviewId")

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

    def rest_most_popular(self):


        # Restaurantes por popularidad
        rest_popularity = self.DATA["TRAIN"].id_restaurant.value_counts().reset_index().rename( columns={"index": "id_restaurant", "id_restaurant": "n_reviews"})
        res = rest_popularity.iloc[0].to_list()
        res.append(res[-1]/self.DATA["N_USR"])

        print(self.CONFIG["city"]+"\t"+"\t".join(map(lambda x: str(x), res)))

    def dataset_stats(self):

        print(self.CONFIG["city"])
        print("#Rest\t#Users\t#Reviews\t#Images")
        print(str(len(self.DATA["TRAIN"].restaurantId.unique()))+"\t"+str(len(self.DATA["TRAIN"].userId.unique()))+"\t"+str(len(self.DATA["TRAIN"]))+"\t"+str(self.DATA["TRAIN"].num_images.sum()))
        print(str(len(self.DATA["TEST"].restaurantId.unique()))+"\t"+str(len(self.DATA["TEST"].userId.unique()))+"\t"+str(len(self.DATA["TEST"]))+"\t"+str(self.DATA["TEST"].num_images.sum()))

