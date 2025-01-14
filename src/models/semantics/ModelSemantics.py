# -*- coding: utf-8 -*-

from src.Common import print_e

import sys
from tqdm import tqdm
import pandas as pd

from src.models.ModelClass import *
from scipy.spatial.distance import cdist

from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, save

from sklearn.manifold import TSNE

########################################################################################################################

class ModelSemantics(ModelClass):

    def get_model(self):
        raise NotImplementedError

    def train(self, save=False):
        raise NotImplementedError

    def __init__(self,config, dataset):
        ModelClass.__init__(self, config=config, dataset=dataset)

    def test(self, encoding="", metric="dist", n_relevant=1 , previous_result=None, log2file=False):

        def getRelevantRestaurants(number, current, others, others_data, relevant):

            if "dist" in metric:
                dists = cdist([current], others)[0]
                arg_dist_sort = np.argsort(dists).flatten()

            elif "dot" in metric:
                dists = [np.dot(current, others[i, :]) for i in range(len(others))]
                arg_dist_sort = np.argsort((np.asarray(dists) * -1))
            else:
                return Exception

            all_rsts_ordered = list(dict.fromkeys(np.asarray(others_data)[arg_dist_sort]))
            n_fts = number  # Número de fotos más cercanas
            ret = []

            while len(ret) < number:
                idxs =arg_dist_sort[:n_fts]
                ret = list(dict.fromkeys(np.asarray(others_data)[idxs]))  # Eliminar duplicados conservando el orden
                n_fts += 1

            rlvnt_pos = {}

            for rlv in relevant:
                tmp = np.argwhere(all_rsts_ordered == rlv).flatten()[0]
                rlvnt_pos[rlv] = tmp

            first_pos = np.min(list(rlvnt_pos.values()))


            return ret, n_fts - 1, idxs, first_pos

        #Almacenar la salida en fichero
        log_file_path = self.MODEL_PATH+"test.txt"
        if log2file and os.path.exists(log_file_path): print_e("Test already done"); exit(0)
        if log2file: sys.stdout = open(log_file_path, "w")

        #print("*"*100)
        #print(self.DATASET.CONFIG["city"], self.CONFIG["model"], encoding)
        #print("*"*100)

        # Cargar los datos de los usuarios apartados al principio (FINAL_USRS)
        FINAL_USRS = self.DATASET.DATA["TEST"]
        FINAL_USRS = FINAL_USRS.merge(self.DATASET.DATA["IMG"], on="reviewId")

        all_img_embs = self.__get_image_encoding__(encoding=encoding) # Obtener embeddings de todas las imágenes
        all_img_rest = self.DATASET.DATA["IMG"].id_restaurant.to_list()

        train_img_embs = np.delete(all_img_embs, FINAL_USRS.id_img.unique(), 0) # Obtener embeddings de imágenes utilizadas en entrenamiento
        train_img_rest = self.DATASET.DATA["IMG"].loc[self.DATASET.DATA["IMG"].test==False].id_restaurant.to_list()
        train_img_url = self.DATASET.DATA["IMG"].loc[self.DATASET.DATA["IMG"].test==False].url.to_list()

        #################################################################################

        ret = []
        rest_rec = []
        rest_rel = []

        for id, r in tqdm(FINAL_USRS.groupby("userId")):
            uid = r["userId"].values[0]
            relevant = r["id_restaurant_y"].unique()

            n_revs = len(r.reviewId.unique())
            n_imgs = len(r)

            img_idxs = r.id_img.to_list()
            mean_img = np.mean(all_img_embs[img_idxs], axis=0)

            #imgs_url = self.DATASET.DATA["IMG"].iloc[img_idxs].url.values

            retrieved, n_m, imgs, first_pos = getRelevantRestaurants(n_relevant, mean_img, train_img_embs, train_img_rest, relevant)

            acierto = int(first_pos<n_relevant)

            rest_relevant = r.loc[r.id_restaurant_y.isin(relevant)].rest_name.unique().tolist()
            rest_retrieved = self.DATASET.DATA["TRAIN"].loc[self.DATASET.DATA["TRAIN"].id_restaurant.isin(retrieved)].rest_name.unique().tolist()
            img_relevant = img_idxs
            img_retrieved = list(imgs)

            rest_rec.extend(retrieved)
            rest_rel.extend(relevant)

            intersect = len(set(retrieved).intersection(set(relevant)))

            prec = intersect / len(retrieved)
            rec = intersect / len(relevant)

            f1 = 0
            if(prec>0 or rec>0):
                f1 = 2*((prec*rec)/(prec+rec))

            ret.append((uid, first_pos, acierto, n_revs, n_imgs, prec, rec, f1, n_m, rest_relevant, rest_retrieved, img_relevant, img_retrieved))

        ret = pd.DataFrame(ret, columns=["user","first_pos","acierto","n_revs","n_imgs","precision","recall","F1","#recov","rest_relevant","rest_retrieved","img_relevant","img_retrieved"])

        ret.to_excel("docs/"+encoding.lower()+"_test.xlsx")

        pr = ret["precision"].mean()
        rc = ret["recall"].mean()
        f1 = ret["F1"].mean()

        print("%f\t%f\t%f\t%f\t%f" % (pr,rc,f1,ret["#recov"].mean(),ret["#recov"].std()))
        print("%d\t%f" % (ret.acierto.sum(),ret.acierto.sum()/ret.acierto.count()))
        print("%f\t%f\t%f" % (ret.first_pos.mean(),  ret.first_pos.median(),ret.first_pos.std()))


        # Desglosar resultados por número de restaurantes

        ret["n_rest"] = ret.rest_relevant.apply(lambda x: len(x))

        desglose = []

        for n_r, rdata in ret.groupby("n_rest"):
            desglose.append((n_r, len(rdata), rdata["first_pos"].median(), rdata["F1"].mean(), rdata["acierto"].sum()))
            #print("%d\t%d\t%f\t%f\t%f" % (n_r, len(rdata),rdata["first_pos"].median(), rdata["F1"].mean(), rdata["acierto"].sum()))

        desglose = pd.DataFrame(desglose, columns=["n_rsts","n_casos","median","f1","aciertos"])

        desglose["n_casos_sum"] = (desglose["n_casos"].sum() - desglose["n_casos"].cumsum()).shift(1, fill_value=desglose["n_casos"].sum())
        desglose["aciertos_sum"] = (desglose["aciertos"].sum()-desglose["aciertos"].cumsum()).shift(1, fill_value=desglose["aciertos"].sum())
        desglose["prctg"] = desglose["aciertos_sum"] / desglose["n_casos_sum"]

        print("·"*100)

        for i in [1,2,3,4]:
            if desglose.n_rsts.min()> i: continue
            tmp = desglose.loc[desglose.n_rsts == i][["n_casos_sum", "prctg"]]
            print("%d\t%d\n%f" % (i,tmp.n_casos_sum.values[0],tmp.prctg.values[0]))

        #desglose.to_clipboard(excel=True)

        if previous_result== None:
            previous_result = {encoding:ret}
        else:
            previous_result[encoding] = ret



        # Cerrar fichero de salida
        if log2file: sys.stdout.close()

        return previous_result

    def emb_tsne(self,layer_name="img_emb"):

        show_rests = {"V. Crespo": {"color": "orange"},
                      "Gloria Gijón": {"color": "teal"},
                      "Restaurante Auga": {"color": "yellow"},
                      "La Salgar": {"color": "magenta"},
                      "Tierra Astur Poniente": {"color": "darkgreen"},
                      "SIDRERÍA CASA CARMEN": {"color": "crimson"},
                      "Restaurante Sidreria La Galana": {"color": "blue"},
                      "Dosmasuno Gastro": {"color": "pink"}}

        show_rests = {"A Feira Do Pulpo": {"color": "orange"},
                      "El Lavaderu": {"color": "teal"},
                      "La Pondala": {"color": "yellow"},
                      "La Salgar": {"color": "magenta"},
                      "El Refugio": {"color": "darkgreen"},
                      "Restaurante Ume": {"color": "crimson"},
                      "Restaurante El Centenario": {"color": "blue"}
                      }

        show_rests = {"Tierra Astur Poniente": {"color": "magenta"}}

        # ----------------------------------------------------------------------------------------------------------------------

        out_path = "/var/www/html/bokeh/"+self.CUSTOM_PATH
        os.makedirs(out_path, exist_ok=True)
        output_file(out_path + "plot.html")

        IMG_DATA = self.DATASET.DATA["IMG"]
        IMG_DATA = IMG_DATA.merge(self.DATASET.DATA["TRAIN"][["id_restaurant", "rest_name"]].drop_duplicates())

        imgs = IMG_DATA.url.to_list()
        imgs_id = IMG_DATA.id_img.to_list()
        colors = np.asarray(["lightgray"] * len(imgs))
        alpha = np.asarray([1] * len(imgs))
        rsts_id = IMG_DATA.id_restaurant.to_list()
        rsts = IMG_DATA.rest_name.to_list()

        for i, k in enumerate(show_rests.keys()):
            idxs = np.argwhere(np.asarray(rsts) == k).flatten()
            colors[idxs] = show_rests[k]["color"]
            alpha[idxs] = 1.

        # las de test, marcarlas de alguna forma
        tst_idxs = IMG_DATA.loc[IMG_DATA.test == True].id_img.to_list()
        colors[tst_idxs] = "black"

        # las del padrino de test, marcarlas de alguna forma
        pad_idxs = IMG_DATA.loc[(IMG_DATA.test == True) & (IMG_DATA.rest_name == list(show_rests.keys())[0])].id_img.to_list()
        colors[pad_idxs] = "orange"

        embs_o = self.__get_image_encoding__(encoding="emb", layer_name=layer_name) # Obtener embeddings de todas las imágenes

        if embs_o.shape[1] > 2:
            embs = TSNE(n_components=2, verbose=1, random_state=self.CONFIG["seed"]).fit_transform(embs_o)

        else:
            embs = embs_o

        source = ColumnDataSource(
            data=dict(
                x=embs[:, 0],
                y=embs[:, 1],
                color=colors,
                rest=rsts,
                rest_id=rsts_id,
                img=imgs,
                img_id=imgs_id,
                alpha=alpha,
            )
        )

        hover = HoverTool(

            tooltips="""
                   <div style="display:flex">
                       <div>
                           <img
                               src="@img"  width="150"
                               style="float: left; margin: 0px 15px 15px 0px;"
                               border="2"
                           ></img>
                       </div>
                       <div>
                           <span style="font-size: 15px; font-weight: bold;">[@rest_id] @rest</span>
                           <span>@img_id</span>
                       </div>
                   </div>
                   """
        )

        p = figure(plot_width=1800, plot_height=900, tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"], output_backend="webgl", title=self.MODEL_NAME)
        p.circle('x', 'y', color="color", fill_alpha="alpha", line_alpha="alpha", size=7, source=source)

        save(p)


