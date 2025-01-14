# -*- coding: utf-8 -*-

import nvgpu
import argparse
import re

import os, json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.datasets.semantics.cluster_explantion.OnlyFoodClusterExplanation import *
from src.models.semantics.cluster_explanation.SemPicClusterExplanation import *

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics import pairwise_distances


######################################################################################################################################################
# CONFIGURACIÓN
######################################################################################################################################################

# Obtener argumentos por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, help="city")
parser.add_argument('-m', type=str, help="model_version")
parser.add_argument('-p', type=float, help="pctg")
parser.add_argument('-lr', type=float, help="lrate")
parser.add_argument('-bs', type=int, help="batch")
parser.add_argument('-s', type=int, help="stage")
parser.add_argument('-csm', type=str, help="clusters_selection_mode")
parser.add_argument('-inc', type=int, help="img_n_clusters")
parser.add_argument('-ism', type=int, help="Image selection model (0/1) (clustering/random)")
parser.add_argument('-onlypos', type=int, help="Only positive reviews (0/1) (False/True)")

args = parser.parse_args()

rst_clustering_distance = "dot"  # "cos"
best_clusters_selection_mode = "m10" if args.csm is None else args.csm
img_n_clusters = 5 if args.inc is None else args.inc
image_selection_mode = 0 if args.ism is None else args.ism

city = "gijon".lower().replace(" ", "") if args.c is None else args.c

seed = 100
active_usrs = True
gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
l_rate = 5e-4 if args.lr is None else args.lr
n_epochs = 4000  # 6000
b_size = 1024 if args.bs is None else args.bs
m_name = "hl5" if args.m is None else args.m
only_positives = 0 if args.onlypos is None else args.onlypos # 0 todos 1 solo positivos

stage = 2 if args.s is None else args.s

######################################################################################################################################################
# MÉTODOS
######################################################################################################################################################

def create_data(dts_inner, mdl_inner):

    # Obtener los datos utilizados para el entrenamiento IMG->Lista de usuarios
    train_data_inner = dts_inner.DATA['TRAIN']
    mdl_seq = mdl_inner.Sequence_DEV(mdl_inner)

    # Eliminar repetidos del conjunto anterior y quedarnos solo con los vectores de usuarios de cada restaurante
    restaurant_data_inner = mdl_seq.ALL_DATA.copy()
    rst_n_imgs = restaurant_data_inner.groupby("id_restaurant").apply(lambda x: pd.Series({"num_imgs": len(x)})).reset_index()
    restaurant_data_inner = restaurant_data_inner[["id_restaurant", "rest_name", "output"]].drop_duplicates("id_restaurant").reset_index(drop=True)
    restaurant_data_inner["n_usrs"] = restaurant_data_inner.output.apply(lambda x: len(x))
    restaurant_data_inner = restaurant_data_inner.merge(rst_n_imgs)

    # Transformar a khot los vectores de usuarios
    restaurant_y_inner = mdl_seq.KHOT.fit_transform(restaurant_data_inner.output.values)

    return train_data_inner, restaurant_data_inner, restaurant_y_inner


def create_data_old(dts_inner, pctg_usrs_inner, remove_zeros=True):
    # Crear ids de usuarios en función de su actividad
    train_data_inner = dts_inner.DATA['TRAIN'].copy()

    usr_list = train_data_inner.groupby("id_user").like.count().sort_values(ascending=False).reset_index().rename(columns={"like": "new_id_user"})

    '''
    # Usar solo usuarios que tengan entre 2 y 15 reviews, los otros son raros
    usr_list["valid"] = usr_list.new_id_user.apply(lambda x: False if (x < 2 or x > 15) else True)
    new_ids = dict(zip(usr_list.loc[usr_list.valid == True].id_user.values, range(len(usr_list.loc[usr_list.valid == True]))))
    usr_list["new_id_user"] = usr_list.id_user.apply(lambda x: int(new_ids[x]) if x in new_ids.keys() else np.inf)
    '''

    usr_list["new_id_user"] = list(range(len(usr_list)))
    train_data_inner = train_data_inner.merge(usr_list).drop(columns=["id_user"]).rename(columns={"new_id_user": "id_user"})

    # Para cada restaurante, obtener la lista de usuarios
    n_usrs = int(dts_inner.DATA["N_USR"] * pctg_usrs_inner)
    khot = MultiLabelBinarizer(classes=list(range(n_usrs)))
    restaurant_data_inner = []

    for id_r, rows in train_data_inner.groupby("id_restaurant"):
        rltd_usrs = rows.id_user.unique()
        rltd_usrs = rltd_usrs[np.argwhere(rltd_usrs < n_usrs).flatten()]
        restaurant_data_inner.append((id_r, rows.rest_name.values[0], rltd_usrs.tolist(), rows.num_images.sum()))

    # Obtener la lista de usuarios en formato KHOT
    restaurant_data_inner = pd.DataFrame(restaurant_data_inner, columns=["id_restaurant", "rest_name", "output", "num_imgs"])
    restaurant_data_inner["n_usrs"] = restaurant_data_inner.output.apply(lambda x: len(x))

    if remove_zeros: restaurant_data_inner = restaurant_data_inner.loc[restaurant_data_inner.n_usrs > 0].reset_index(drop=True)  # Quitar los restaurantes sin usuario

    restaurant_y_inner = khot.fit_transform(restaurant_data_inner.user_list)

    return train_data_inner, restaurant_data_inner, restaurant_y_inner


def restaurant_clustering(restaurant_data_inner, restaurant_y_inner, rst_clustering_distance_inner, best_clusters_selection_mode_inner):
    # Clustering Aglomerativo con coseno utilizando percentil
    # Tirada con dot/cos linkage=complete y umbral que de numerinos wapos (5/95)

    if "dot" in rst_clustering_distance_inner:
        print_g("Restaurant distances...")
        all_distances = pdist(restaurant_y_inner, metric=np.dot)  # DISTANCIA
        print_g("Restaurant distances done!")
        treshold = np.percentile(all_distances.flatten(), 5)
        print_g("Restaurant clustering...")
        clustering = AgglomerativeClustering(n_clusters=None, affinity=lambda X: -pairwise_distances(X, metric=np.dot), linkage="complete",
                                             compute_full_tree=True, distance_threshold=-treshold)
        clustering.fit(restaurant_y_inner)
        print_g("Restaurant clustering done!")

    # Filtrar clusters
    _, sz = np.unique(clustering.labels_, return_counts=True)

    if "u" in best_clusters_selection_mode_inner:
        # Los x con más usuarios
        sz_value = int(best_clusters_selection_mode_inner.replace("u", ""))
        best_clusters = restaurant_data_inner.copy()
        best_clusters["cluster"] = clustering.labels_
        best_clusters = best_clusters.groupby("cluster").apply(lambda x: sum(x.n_usrs)).reset_index() \
                            .sort_values(0, ascending=False).cluster.values[:sz_value]
        rp = dict(zip(best_clusters, range(len(best_clusters))))
    elif "m" in best_clusters_selection_mode_inner:
        # Los x con más Nº usuarios medio
        sz_value = int(best_clusters_selection_mode_inner.replace("m", ""))
        best_clusters = restaurant_data_inner.copy()
        best_clusters.insert(0, "cluster", clustering.labels_, False)
        best_clusters = best_clusters.groupby("cluster").apply(lambda x: pd.Series(
            {"ratio": sum(x.n_usrs) / len(x), "n_users": sum(x.n_usrs), "n_rst": len(x),"rst_names": np.unique(x.rest_name),  "n_imgs": sum(x.num_imgs)})).reset_index().sort_values(
            "ratio", ascending=False)
        best_clusters.insert(1, "new_cluster", range(len(best_clusters)), False)

        print("-" * 100)
        print(best_clusters.head(sz_value))

        rp = dict(zip(best_clusters.cluster.to_list()[:sz_value], best_clusters.new_cluster.to_list()[:sz_value]))

    else:
        # Los x más grandes en
        sz_value = int(best_clusters_selection_mode_inner)
        best_clusters = np.argsort(-sz)[:sz_value]
        rp = dict(zip(best_clusters, range(len(best_clusters))))

    print("-" * 100)
    print( "%s - %s [%d]" % (city,best_clusters_selection_mode_inner,img_n_clusters ))
    print("-" * 100)

    restaurant_data_inner["cluster"] = list(map(lambda x: rp[x] if x in rp.keys() else -1, clustering.labels_))

    cluster_data_inner = restaurant_data_inner.groupby("cluster") \
        .apply(lambda x: np.unique(np.concatenate(x.output.to_list()))).reset_index().rename(columns={0: "restaurants"})

    return restaurant_data_inner, cluster_data_inner


def test_restaurant_clustering(data, rst_data, rst_y):
    user_data = []

    # Datos de los restaurantes que están en los clusters seleccionados
    sel_rst_data = rst_data.loc[rst_data.cluster >= 0]
    sel_rst_data_y = rst_y[sel_rst_data.index.values]
    sel_rst_data_cl = sel_rst_data.cluster.values

    for u, udata in data.groupby("userId"):  # Para cada usuario

        rst_lst = udata[["id_restaurant", "num_images"]]

        if len(rst_lst) > 1:

            img_dist = []

            for _, rdata in rst_lst.iterrows():
                crst_y = rst_data.loc[rst_data.id_restaurant == rdata.id_restaurant]
                if len(crst_y) > 0: # Si el restaurante no está en train, saltar
                    crst_y = rst_y[crst_y.index[0]]
                    crst_dists = cdist(np.expand_dims(crst_y, 0), sel_rst_data_y, metric=np.dot).flatten()
                    slc_cstr = sel_rst_data_cl[crst_dists.argmax()]
                    # slc_cstr = pd.DataFrame(zip(sel_rst_data_cl, crst_dists), columns=["cls", "dst"]).groupby("cls").apply(lambda x: sum(x.dst)).argmax()
                    img_dist.append((slc_cstr, rdata.num_images))

            if len(img_dist) == 0:
                continue  # Si todos los restaurante tienen to-do 0s, saltar?

            img_dist = pd.DataFrame(img_dist, columns=["cluster", "num_images"])
            n_imgs = img_dist.num_images.sum()
            img_dist = img_dist.groupby("cluster").apply(lambda x: sum(x.num_images)).reset_index()
            img_dist["img_dist_prob"] = (img_dist[0].values) / n_imgs

            clusters = img_dist.cluster.values

            unct = lambda p: 0 if p == 0 else (p * np.log2(p))  # incertidumbre
            uncert = -np.sum(list(map(unct, img_dist["img_dist_prob"]))) * n_imgs

            user_data.append((u, clusters, img_dist[0].values, img_dist["img_dist_prob"].values, uncert, n_imgs,
                              len(rst_lst), len(np.unique(clusters))))

    user_data = pd.DataFrame(user_data,
                             columns=["userId", "cluster", "img_dist", "img_dist_prob", "uncert", "n_imgs", "n_rsts",
                                      "n_clusters"])

    return user_data


def split_results(data, distances=False):
    # >=
    for t in [1, 2, 3, 4]:
        tmp_data = data.loc[data.n_rsts >= t]
        if distances:
            print(">=%d\t%d\t%f\t%f" % (t, len(tmp_data), tmp_data.uncert.sum() / tmp_data.n_imgs.sum(), tmp_data.avg_dist.mean()))
        else:
            print(">=%d\t%d\t%f" % (t, len(tmp_data), tmp_data.uncert.sum() / tmp_data.n_imgs.sum()))

    print("-" * 100)

    '''
    # ==
    for t in np.sort(data.n_rsts.unique()):
        tmp_data = data.loc[data.n_rsts == t]
        print("%d\t%d\t%f" % (t, len(tmp_data), tmp_data.uncert.mean()))

    print("-" * 100)
    '''


def get_selected_images(img_n_clusters_in, img_emb_data, random_images= False, debug= False):

    def tsne_and_selected_plot(selected_idxs, suffix=""):

        path = "out/cluster_explanation/tsne/%s/" % city

        os.makedirs(path, exist_ok=True)

        selected = np.zeros(len(cl_rst_img_data))
        selected[selected_idxs] = 1

        tsne = TSNE(2, random_state=seed).fit_transform(cl_rst_img_embs[e]["data"])
        splt = sns.scatterplot(tsne[:, 0], tsne[:, 1], alpha=.25)
        splt = sns.scatterplot(tsne[[selected_idxs], 0][0], tsne[[selected_idxs], 1][0], alpha=1)
        name = "%s-%s-%d-%s" % (city, e, cid, suffix)
        splt.set_title(name)
        plt.savefig(path+name+".jpg")
        plt.clf()

    if random_images: print_w("Selecting images randomly...")

    (densenet_in, yHat_in, img_urls_in) = img_emb_data

    selected_images_inner = []
    rst_img_clst_data = []  # Almacenar datos de los clusters de imágenes que se realizan dentro de los clusters de usuarios.

    for cid, c_data in restaurant_data.groupby("cluster"):  # PARA CADA CLUSTER DE RESTAURANTES
        if cid < 0: continue
        # print("Cluster %d" % (cid+1))

        # Datos de los restaurantes del cluster actual
        c_rsts = c_data.id_restaurant.values
        cl_rst_img_data = dts.DATA['IMG'].loc[(dts.DATA['IMG'].test == False) & (dts.DATA['IMG'].id_restaurant.isin(c_rsts))].reset_index(drop=True)
        cl_rst_img_ids = cl_rst_img_data.id_img.values
        cl_rst_img_embs = {"densenet": {"data": densenet_in[cl_rst_img_ids], "affinity": "euclidean", "metric": "euclidean", "closest": np.argmin, "percentile": 95, "tresh_sign": 1},
                    # "rk": {"data": yHat_in[cl_rst_img_ids], "affinity": "euclidean", "metric": "euclidean", "closest": np.argmin}}
                    # "rk": {"data": yHat_in[cl_rst_img_ids], "affinity": lambda X: -pairwise_distances(X, metric=np.dot), "metric": np.dot, "closest": np.argmax, "percentile": 5, "tresh_sign": -1}}
                      "rk": {"data": yHat_in(cl_rst_img_ids), "affinity": lambda X: -pairwise_distances(X, metric=np.dot), "metric": np.dot, "closest": np.argmax, "percentile": 5, "tresh_sign": -1}}

        cl_rst_img_urls = img_urls_in[cl_rst_img_ids]

        # print(cid, len(c_data) ,c_data.n_usrs.sum(),len(cl_rst_img_ids))

        for e in cl_rst_img_embs.keys():  # PARA CADA TIPO DE CODIFICACIÓN
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
                rst_img_clst_data.append((cid, e, -np.sort(-itms)[:img_n_clusters], rst_dist))

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
                rnd_imgs = cl_rst_img_data.sample(img_n_clusters_in, replace=False, random_state=seed)
                _n = img_n_clusters_in

                if debug: tsne_and_selected_plot(rnd_imgs.index.values, "5 random")

                selected_images_inner.extend(list(zip([cid]*_n, [e]*_n, range(5), [-1]*_n, [cl_rst_img_data.id_restaurant.unique()]*_n, [(0, 0, 0)]*_n, cl_rst_img_embs[e]["data"][rnd_imgs.index.values], rnd_imgs.url.values)))

    selected_images_inner = pd.DataFrame(selected_images_inner, columns=["cluster", "encoding", "image_cluster", "cl_img_rst_lst", "cl_rst_rst_lst", "distances", "img_emb", "img_url"])

    if not random_images and debug:
        rst_img_clst_data = pd.DataFrame(rst_img_clst_data, columns=["rst_cluster", "encoding", "n_img_cls", "n_rst_cls"])

        plot_img_dist= lambda x: sns.heatmap(np.row_stack(rst_img_clst_data.loc[rst_img_clst_data.encoding == x].n_img_cls), annot=True, fmt="d", cmap='Greens').set_title("img dist "+x)
        plot_rst_dist= lambda x: sns.heatmap(np.row_stack(rst_img_clst_data.loc[rst_img_clst_data.encoding == x].n_rst_cls), annot=True, fmt="d", cmap='Greens').set_title("rst dist "+x)

        plot_img_dist("densenet");plt.show()
        plot_img_dist("rk");plt.show()
        plot_rst_dist("densenet");plt.show()
        plot_rst_dist("rk");plt.show()

    return selected_images_inner


def test_selected_images(data, selected_images, img_embs, rst_in_train, suffix="test"):

    for e, e_data in selected_images.groupby("encoding"):  # Primero las 50 de una codificación y luego las 50 de la otra

        e_data_emb = np.row_stack(e_data.img_emb) # Imágenes representativas

        user_data = []

        # Para cada usuario, obtener la distancia mínima de sus imágenes y las represetnativas
        for u, udata in tqdm(data.groupby("userId"), desc=e):

            # Verificar que alguno de los restaurantes del usuario está en los seleccionados en el clustering de restaurantes.
            assert any([x in rst_in_train for x in udata.id_restaurant])

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
            usr_rst_in_clst_rst = int(any([x in selected_cluster_rsts for x in udata.id_restaurant.values]))

            user_data.append((u,selected_cluster, len(udata.id_restaurant), min_dists.iloc[selected_idx].dist, usr_rst_in_clst_rst))

        user_data = pd.DataFrame(user_data, columns=["userId", "cluster", "n_rsts", "dist", "usr_rst_in_clst_rst"])

        for t in [1, 2, 3, 4]:
            tmp_data = user_data.loc[user_data.n_rsts >= t]
            print(">=%d\t%d\t%d\t%f" % (t, len(tmp_data), tmp_data.usr_rst_in_clst_rst.sum(), tmp_data.usr_rst_in_clst_rst.sum() / len(tmp_data)))

        print("-" * 100)


def test_baseline(data, rst_data, rst_in_train):

    """ Para cada usuario, asignarle siempre el cluster de restaurantes con más imágenes """
    # Buscar el cluster de restaurantes con más imágenes
    rst_data = rst_data.loc[rst_data.cluster >= 0]
    selected_cluster = rst_data.groupby("cluster")["num_imgs"].sum().argmax()
    selected_cluster_rsts = rst_data.loc[rst_data.cluster == selected_cluster].id_restaurant.values

    user_data = []

    # Para cada usuario de TEST
    for u, udata in tqdm(data.groupby("userId"), desc="baseline"):
        # Verificar que alguno de los restaurantes del usuario está en los seleccionados en el clustering de restaurantes.
        assert any([x in rst_in_train for x in udata.id_restaurant])

        # Ver si alguno de los restaurantes a los que fue el usuario está en los del cluster seleccionado
        usr_rst_in_clst_rst = int(any([x in selected_cluster_rsts for x in udata.id_restaurant.values]))

        user_data.append((u, selected_cluster, list(udata.rest_name.values), len(udata.id_restaurant), -1, usr_rst_in_clst_rst))

    user_data = pd.DataFrame(user_data, columns=["userId", "cluster", "rest_names", "n_rsts", "dist", "usr_rst_in_clst_rst"])
    
    # user_data.to_excel("test_user_data_%s.xlsx" % ("pos" if only_positives else "all"))

    for t in [1, 2, 3, 4]:
        tmp_data = user_data.loc[user_data.n_rsts >= t]
        print(">=%d\t%d\t%d\t%f" % (t, len(tmp_data), tmp_data.usr_rst_in_clst_rst.sum(), tmp_data.usr_rst_in_clst_rst.sum() / len(tmp_data)))

    print("-" * 100)

    exit()


def collect_gridsearch_results(city, out_col="val_f1", monitor="val_f1"):
    path = "models/SemPicClusterExplanation/%s/" % city
    ret = []
    for f in os.listdir(path):
        exp_path = path+f+"/"
        cfg_file = exp_path+"cfg.json"
        csv_file = exp_path+"log.csv"

        if os.path.exists(csv_file) and os.path.exists(cfg_file):
            with open(cfg_file) as json_file: cfg = json.load(json_file)
            train_history = pd.read_csv(csv_file)
            train_history_best = train_history.iloc[train_history[monitor].argmax()]
            ret.append(np.concatenate([list(cfg["model"].values()), [cfg["data_filter"]["pctg_usrs"]], [train_history_best[out_col]]]))

    ret = pd.DataFrame(ret, columns=np.concatenate([list(cfg["model"].keys()), ["pctg_usrs"], [out_col]]))
    ret.learning_rate = ret.learning_rate.astype(float)
    ret.loc[ret.model_version == "hl5", "model_version"] = "hl05"
    ret = ret.loc[ret.model_version.str.contains("hl")].sort_values(["model_version", "pctg_usrs"]) # MUY IMPORTANTE EL ORDEN

    multi_column = list(zip(np.repeat(ret.model_version.unique(),2), np.tile(ret.pctg_usrs.unique(), len(ret.model_version.unique()))))
    index = pd.MultiIndex.from_tuples(multi_column, names=["model_version", "pctg_usrs"])
    print_data = pd.DataFrame([], columns=index)
    _p_d = pd.DataFrame([], columns=["learning_rate"])
    print_data = pd.concat([_p_d, print_data], axis=1)

    for lr, lr_data in ret.groupby("learning_rate"):
        tmp = lr_data.copy()
        tmp = tmp.sort_values(["model_version", "pctg_usrs"])

        tmp_app_data = dict(zip(list(zip(tmp.model_version,tmp.pctg_usrs)),tmp[out_col]))
        tmp_app_data["learning_rate"] = lr

        print_data = print_data.append(tmp_app_data, ignore_index=True)

    print_data = print_data.sort_values("learning_rate", ascending=False)
    print(print_data.to_csv(sep="\t", index=False))


def collect_test_results(city, only_positives=False):

    base_path = "/home/pperez/PycharmProjects/SemPic/out/cluster_explanation/test/%s/%d/" % (city, bool(only_positives))

    print(base_path)

    res = []

    for f in os.listdir(base_path):

        if ".xlsx" in f: continue

        rsts, imgs, rnd = re.findall(r"m(\d+)-(\d+)-\[(\d)\]", f)[0]
        rsts = int(rsts); imgs = int(imgs); rnd = int(rnd)
        with open(base_path+f, "r") as file: content = file.readlines()

        bl = pd.DataFrame(list(map(lambda x: x.strip().split("\t"), content[-15:-11])))
        dn = pd.DataFrame(list(map(lambda x: x.strip().split("\t"), content[-10:-6])))
        yh = pd.DataFrame(list(map(lambda x: x.strip().split("\t"), content[-5:-1])))

        res.append((rsts, imgs, rnd, "bl", bl.iloc[0, 3], bl.iloc[1, 3], bl.iloc[2, 3], bl.iloc[3, 3]))
        res.append((rsts, imgs, rnd, "dn", dn.iloc[0, 3], dn.iloc[1, 3], dn.iloc[2, 3], dn.iloc[3, 3]))
        res.append((rsts, imgs, rnd, "yh", yh.iloc[0, 3], yh.iloc[1, 3], yh.iloc[2, 3], yh.iloc[3, 3]))

    res = pd.DataFrame(res, columns=["n_rst", "n_imgs", "rnd", "emb", ">=1", ">=2", ">=3", ">=4"])

    for e in ["bl", "dn", "yh"]:
        for r in [0,1]:
            print(e,r)
            tmp_res = res.loc[(res["emb"] == e) & (res["rnd"] == r)]
            for _, g in tmp_res.groupby("n_rst"):
                line = "\t".join(g.sort_values("n_imgs")[">=1"])
                print("%d\t%s" % (_,line))

    print("-"*20)

    # Crear un excel con los resultados desglosados de cada combinación
    res = res.sort_values(["n_rst", "n_imgs", "rnd", "emb"])
    # res[[">=1", ">=2", ">=3", ">=4"]] = res[[">=1", ">=2", ">=3", ">=4"]].applymap(lambda x: str(x).replace(".", ","))

    with pd.ExcelWriter(base_path + 'all_%s.xlsx' % city, engine='xlsxwriter', options={'strings_to_numbers': True}) as writer:

        cell_format = writer.book.add_format()
        cell_format.set_border(1)

        format0 = writer.book.add_format({'align': 'center'})
        format1 = writer.book.add_format({'align': 'center', "num_format": 10})
        format2 = writer.book.add_format({'align': 'center', 'bold': 1, 'fg_color': '#808080', 'border': 1})
        format3 = writer.book.add_format({'align': 'center', 'bold': 1, 'fg_color': '#BFBFBF', 'border': 1})

        for idx, (_, data) in enumerate(res.groupby(["n_rst", "n_imgs"])):
            data.iloc[:3, [0, 1, 3, 4, 5, 6, 7]].to_excel(writer, sheet_name='data', header=False, index=False, startrow=(3*idx)+(1*idx)+3, startcol=1)
            data.iloc[3:, 4:].to_excel(writer, sheet_name='data', header=False, index=False, startrow=(3*idx)+(1*idx)+3, startcol=1+(data.iloc[:3, [0, 1, 3, 4, 5, 6, 7]].shape[1] + 1))

            for col in ["E", "F", "G", "H", "J", "K", "L", "M"]:
                writer.sheets["data"].conditional_format("%s%d:%s%d" % (col, 4+(idx*4), col, 4+(idx*4)+2), {'type': '3_color_scale'})

        writer.sheets["data"].set_column("B:E", None, format0)
        writer.sheets["data"].set_column("E:M", None, format1)
        writer.sheets["data"].set_column("I:I", 1.43)

        writer.sheets["data"].merge_range('B2:H2', 'Clustering', format2)
        writer.sheets["data"].merge_range('J2:M2', 'Random', format2)

        writer.sheets["data"].write_row(2, 1, ["#RST", "#IMG", "EMB", ">=1", ">=2", ">=3", ">=4"], format3)
        writer.sheets["data"].write_row(2, 9, [">=1", ">=2", ">=3", ">=4"], format3)

    exit()

    for emb, emb_data in res.loc[res["n_rst"] == 10].groupby("emb"):
        for imgs, img_data in emb_data.groupby("n_imgs"):
            data_tmp = list(zip(img_data.loc[img_data.rnd == 1][[">=1", ">=2", ">=3", ">=4"]].values[0],img_data.loc[img_data.rnd == 0][[">=1", ">=2", ">=3", ">=4"]].values[0]))
            print("\n".join(map(lambda x: "\t".join(x), data_tmp)))
            print()


######################################################################################################################################################
# 1.- ENTRENAR EL MODELO [IMG->RST(25% de usuarios)]
######################################################################################################################################################

# ToDo: Re-entrenar el modelo para todas las ciudades (solo está para Gijón y BCN)
# ToDo: Revisar todo el proceso mirando la guia del Teams
# ToDo: Los porcentajes del test, son justos?

# ToDo: La división en TRAIN TEST influye mucho en los resultados del baseline
'''
collect_test_results(city, only_positives=True)
exit()
'''

'''
collect_gridsearch_results(city, out_col="val_f1")
collect_gridsearch_results(city, out_col="val_precision")
collect_gridsearch_results(city, out_col="val_recall")
exit()
'''
pctg_usrs = .25 if args.p is None else args.p

data_cfg = {"city": city, "data_path": "/media/HDD/pperez/TripAdvisor/" + city + "_data/", "seed": seed, "only_positives": bool(only_positives)}
data_cfg["save_path"] = data_cfg["data_path"]

dts = OnlyFoodClusterExplanation(data_cfg)  # dts.get_stats()

cfg_u = {"model": {"learning_rate": l_rate, "epochs": n_epochs, "batch_size": b_size, "seed": seed, "model_version": m_name},
         "data_filter": {"active_usrs": bool(active_usrs), "pctg_usrs": pctg_usrs, "min_imgs_per_rest": 5, "min_usrs_per_rest": 5},
         "session": {"gpu": gpu, "in_md5": False}}


print(cfg_u)
print("-"*50)

mdl = SemPicClusterExplanation(cfg_u, dts)

'''
n_usrs = len(mdl.DATASET.DATA["TRAIN"].userId.unique())
n_rst = len(mdl.DATASET.DATA["TRAIN"].restaurantId.unique())
n_rvws = len(mdl.DATASET.DATA["TRAIN"])
n_photos = mdl.DATASET.DATA["TRAIN"].num_images.sum()
print("\t".join(list(map(str, [n_usrs, n_rst, n_rvws, n_photos]))))

exit()
'''

if stage == 0:
    mdl.train_dev(save_model=True)
    # mdl.dev_stats()
    exit()

elif stage == 1:
    mdl.train(save_model=True)
    exit()

######################################################################################################################################################
# 2.- Clustering de restaurantes
######################################################################################################################################################

# Leer/crear datos para el clustering de restaurantes (para cada restaurante, vector de usuarios del 25% que fueron)
train_data, restaurant_data, restaurant_y = create_data(dts_inner=dts, mdl_inner=mdl)


# Clustering (aglomerativo, distancia coseno, percentil 5) de restaurantes con Y de TRAIN utilizando el 50% de los usuarios más activos.
restaurant_data, clustering_data = restaurant_clustering(restaurant_data_inner=restaurant_data,
                                                         restaurant_y_inner=restaurant_y,
                                                         rst_clustering_distance_inner=rst_clustering_distance,
                                                         best_clusters_selection_mode_inner=best_clusters_selection_mode)

######################################################################################################################################################
# 2.1.- Adecuar conjunto de test
######################################################################################################################################################
# Con usuarios de TEST, (>=2 rests) ver en que cluster cae cada uno de sus restaurantes. Si uno de sus restaurantes
# tiene 7 fotos y cae en el cluster 1, se cuenta como 7 elementos que caen en ese cluster, no solo uno.
# Hay que calcular incertidumbre/entropía por usuario ponderada por número de fotos y luego obtener un valor global
# para el conjunto de TEST. Este valor es lo mejor que se puede obtener.
######################################################################################################################################################

# Los de test con imágenes y cuyos restaurantes (alguno) aparezcan en los X clusters seleccionados
test_data = dts.DATA["TEST"].copy()
test_data = test_data.loc[test_data.num_images > 0]
rst_in_train = restaurant_data.loc[restaurant_data.cluster > 0].id_restaurant.unique()

test_usr_rst_train = test_data.groupby("userId").apply(lambda x: int(any([s in x.id_restaurant.values for s in rst_in_train]))).reset_index()
test_usr_rst_train = test_usr_rst_train.loc[test_usr_rst_train[0] > 0].userId.to_list()
test_data = test_data.loc[test_data.userId.isin(test_usr_rst_train)]

'''
print("ONLY POSITIVES:", data_cfg["only_positives"])
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

get_stats(test_data, "FILTERED_TEST")
exit()
'''

# user_data = test_restaurant_clustering(data=test_data, rst_data=restaurant_data, rst_y=restaurant_y)
# split_results(user_data)  # Desglosar resultados por número de restaurantes

######################################################################################################################################################
# 3.- Clustering de imágenes
######################################################################################################################################################
# Buscar fotos representativas: En cada cluster, tenemos la lista de restaurantes, con las fotos de esos restaurantes,
# hacer Kmeans ( k=5y10 dentro del cluster de restaurantes, si) y coger ¿la más cercana al centroide de cada cluster?.
# Otra forma sería hacerlo con Densenet.
######################################################################################################################################################

mdl.MODEL.load_weights(mdl.MODEL_PATH + "weights")
# sub_model = Model(inputs=[mdl.MODEL.get_layer("in").input], outputs=[mdl.MODEL.get_layer("img_emb").output])

densenet = dts.DATA["IMG_VEC"]  # dense

# yHat = lambda x: mdl.MODEL.predict(densenet[x], batch_size=cfg_u["model"]["batch_size"])

def yHat(x):
  with tf.device("cpu:0"):
    return mdl.MODEL.predict(densenet[x], batch_size=cfg_u["model"]["batch_size"])

# with tf.device("cpu:0"): yHat = mdl.MODEL.predict(densenet, batch_size=cfg_u["model"]["batch_size"])  # rk
img_urls = dts.DATA["IMG"].url.values

# selected_images = get_selected_images(img_n_clusters_in=img_n_clusters, img_emb_data=(densenet, yHat, img_urls), random_images=image_selection_mode, debug=False)
selected_images = get_selected_images(img_n_clusters_in=img_n_clusters, img_emb_data=(densenet, yHat, img_urls), random_images=image_selection_mode, debug=False)

######################################################################################################################################################
# 4.- Evaluación
######################################################################################################################################################
# Para evaluar las fotos, se hace lo mismo pero, mirando en que cluster caen las fotos del usuario. ¿Como?
# Ahora en cada cluster, además de una lista de restaurantes, tienes una de fotos representativas, por tanto para cada
# foto del usuario de test, tenemos que mirar el cluster más compatible utilizando el (cos).
# Al finalizar podemos obtener el mismo número que en el punto 2 y comparar.
######################################################################################################################################################

img_embs = {"densenet": {"data": densenet, "metric": "euclidean", "closest": np.argmin},
            # "rk": {"data": yHat, "metric": "euclidean", "closest": np.argmin}}
            # "rk": {"data": yHat, "metric": np.dot, "closest": np.argmax}}
            "rk": {"data": yHat, "metric": np.dot, "closest": np.argmax}}

test_baseline(data=test_data, rst_data=restaurant_data, rst_in_train=rst_in_train)
test_selected_images(data=test_data, selected_images=selected_images, img_embs=img_embs, rst_in_train=rst_in_train)

