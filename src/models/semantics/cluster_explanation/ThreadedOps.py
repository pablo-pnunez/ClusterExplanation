# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import pandas as pd
import numpy as np
import threading
import time

from pandas.core.frame import DataFrame

class ThreadedOps(threading.Thread):


    def __init__(self, threadID, name, counter, data=None, args=None, step=1):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

        self.DATA = data
        self.ARGS = args
        self.STEP = step

        self.TSS = time.time()

        self.RETURN=None


    def run(self):

        # Descargar lista de restaurantes...
        if(self.STEP == 0):
            raise NotImplementedError
        # Seleccionar imágenes para cada cluster de restaurantes...
        elif(self.STEP == 1):
            self.RETURN = self.select_images_restaurant_cluster()

        # Hacer test...
        elif(self.STEP == 2):
            self.RETURN = self.user_test()

    def join(self):
        threading.Thread.join(self)
        # print(f"Thread {self.threadID} end ({time.time()-self.TSS:5.2f}).")
        return self.RETURN

    # -------------------------------------------------------------------------------------------------------------------

    def select_images_restaurant_cluster(self):
        # print(self.name, self.DATA.cluster.min(),self.DATA.cluster.max(), len(self.DATA.cluster.unique()))

        (dts, img_emb_data, random_images, img_n_clusters, seed) = self.ARGS
        (densenet_in, yHat_in, img_urls_in) = img_emb_data

        selected_images_inner = []

        # Para cada cluster de resturantes...
        for cdata in self.DATA.groupby("cluster"):
            cid, c_data = cdata
            
            if cid < 0: return

            # Datos de los restaurantes del cluster actual
            c_rsts = c_data.restaurantId.values
            cl_rst_img_data = dts.DATA['IMG'].loc[(dts.DATA['IMG'].test == False) & (dts.DATA['IMG'].restaurantId.isin(c_rsts))].reset_index(drop=True)

            cl_rst_img_ids = cl_rst_img_data.id_img.values
            cl_rst_img_embs = {
                                "rk": {"data": yHat_in(cl_rst_img_ids), "affinity": lambda X: -pairwise_distances(X, metric=np.dot), "metric": np.dot, "closest": np.argmax, "percentile": 5, "tresh_sign": -1},
                                "densenet": {"data": densenet_in[cl_rst_img_ids], "affinity": "euclidean", "metric": "euclidean", "closest": np.argmin, "percentile": 95, "tresh_sign": 1}
                                }

            cl_rst_img_urls = img_urls_in[cl_rst_img_ids]

            # print(cid, len(c_data) ,c_data.n_usrs.sum(),len(cl_rst_img_ids))

            for e in ["rk", "densenet"]:#cl_rst_img_embs.keys():  # PARA CADA TIPO DE CODIFICACIÓN
                
                # Seleccionar las imágenes representativas mediante clustering
                if not random_images:
                    all_distances = pdist(cl_rst_img_embs[e]["data"], metric=cl_rst_img_embs[e]["metric"])  # DISTANCIA
                    treshold = np.percentile(all_distances.flatten(), cl_rst_img_embs[e]["percentile"])

                    clustering = AgglomerativeClustering(n_clusters=None, affinity=cl_rst_img_embs[e]["affinity"], linkage="complete", compute_full_tree=True, distance_threshold=treshold * cl_rst_img_embs[e]["tresh_sign"])
                    clustering.fit(cl_rst_img_embs[e]["data"])

                    # Seleccionar los X clusters con más imágenes
                    _, itms = np.unique(clustering.labels_, return_counts=True)
                    greatest = np.argsort(-itms)[:img_n_clusters]
                    greatest = dict(zip(range(img_n_clusters), greatest))

                    # Obtener información de los clusters (n_imgs y n_rsts)
                    rst_clusters = pd.DataFrame(zip(cl_rst_img_data.id_restaurant, clustering.labels_), columns=["id_restaurant", "cluster"])
                    rst_clusters = rst_clusters.loc[rst_clusters.cluster.isin(greatest.values())] # Quedarse solo con los restaurantes que aparecen en clusters seleccionados
                    rst_dist = rst_clusters.groupby("cluster").apply(lambda x: len(x.id_restaurant.unique())).values
                    # rst_img_clst_data.append((cid, e, -np.sort(-itms)[:img_n_clusters], rst_dist))

                    cls_slc_imgs = []  # Imágenes seleccionadas para este cluster de restaurantes

                    for k_n in greatest.keys():  # PARA CADA CLUSTER DE IMÁGENES SELECCIONADO
                        k = greatest[k_n]
                        cl_img_img_idxs = np.argwhere(clustering.labels_ == k).flatten()
                        cl_img_img_embs = cl_rst_img_embs[e]["data"][cl_img_img_idxs]
                        cntr = np.mean(cl_img_img_embs, axis=0)
                        cnt_img_distances = cdist(np.expand_dims(cntr, 0), cl_img_img_embs, metric=cl_rst_img_embs[e]["metric"]).flatten()

                        cl_img_rst_lst = rst_clusters.loc[rst_clusters.cluster == k].id_restaurant.unique()

                        inner_sel_idx = cl_rst_img_embs[e]["closest"](cnt_img_distances)  # Index de la imagen seleccionada dentro de cl_img_img_embs (subset de cl_rst_img_data)
                        outer_sel_idx = cl_img_img_idxs[inner_sel_idx]  # Index de la imagen seleccionada pero en cl_rst_img_data (no en cl_img_img_embs, que es un subset)

                        selected_images_inner.append((cid, e, k_n, cl_img_rst_lst, c_rsts, [cnt_img_distances.min(), cnt_img_distances.mean(), cnt_img_distances.max()], cl_img_img_embs[inner_sel_idx], cl_rst_img_urls[outer_sel_idx]))

                # Seleccionar las imágenes representativas de forma aleatoria
                else:
                    rnd_imgs = cl_rst_img_data.sample(img_n_clusters, replace=False, random_state=seed)
                    _n = img_n_clusters

                    selected_images_inner.extend(list(zip([cid]*_n, [e]*_n, range(5), [-1]*_n, [cl_rst_img_data.restaurantId.unique()]*_n, [(0, 0, 0)]*_n, cl_rst_img_embs[e]["data"][rnd_imgs.index.values], rnd_imgs.url.values)))

        return selected_images_inner


    def user_test(self):

        (dts, e, e_data, e_data_emb, img_embs) = self.ARGS

        user_data = []

        for u, udata in tqdm(self.DATA.groupby("userId"), desc=f"Thread {self.threadID:d}"):

            user_images = dts.DATA["IMG"].loc[dts.DATA["IMG"].reviewId.isin(udata.reviewId.values)]
            assert udata.num_images.sum() == len(user_images)

            min_dists = {}

            # Buscar, para cada imagen del usuario, la más cercana de las seleccionadas y su cluster
            for _, i in user_images.iterrows():

                assert dts.DATA["IMG"].iloc[i.id_img].test  # Asegurarse que la imagen es de test

                if e=="rk": emb = img_embs[e]["data"]([i.id_img])[0]
                else: emb = img_embs[e]["data"][i.id_img]

                dists = cdist(np.expand_dims(emb, 0), e_data_emb, metric=img_embs[e]["metric"]).flatten()  # ¿QUE DISTANCIA? Euclidea y dot

                idx_closest_image = img_embs[e]["closest"](dists)
                closest_image_data = e_data.iloc[idx_closest_image]
                close_cluster = closest_image_data.cluster

                min_dists[dists[idx_closest_image]] = close_cluster

            # Seleccionar cual es el cluster más afín al usuario (el menor/mayor de los menores/mayores)
            min_dists = pd.DataFrame(list(zip(min_dists.keys(), min_dists.values())), columns=["dist", "cluster"])
            selected_idx = img_embs[e]["closest"](min_dists.dist)
            selected_cluster = int(min_dists.iloc[selected_idx].cluster)
            selected_cluster_rsts = e_data.loc[e_data.cluster == selected_cluster].cl_rst_rst_lst.values[0]

            # Ver si la alguno de los restaurantes a los que fue el usuario está en los del cluster seleccionado
            usr_rst_in_clst_rst = int(any([x in selected_cluster_rsts for x in udata.restaurantId.values]))

            user_data.append((u,selected_cluster, len(udata.restaurantId), min_dists.iloc[selected_idx].dist, usr_rst_in_clst_rst))
       
        return user_data
