import pandas as pd
import numpy as np 
from scipy.spatial.distance import pdist, squareform

CONFIG={"data_filter":{"min_imgs_per_rest":5}}
ALL = pd.read_pickle("all_data_gjn.pkl")

# Eliminar restaurantes con menos de 5 imágenes
print("- Removing restaurants with less than %d images..." % CONFIG["data_filter"]["min_imgs_per_rest"])
rst_n_img = ALL.groupby("restaurantId").apply(lambda x: pd.Series({"num_images": x.num_images.sum()})).reset_index()
ALL = ALL.loc[ALL.restaurantId.isin(rst_n_img.loc[rst_n_img.num_images >= CONFIG["data_filter"]["min_imgs_per_rest"]].restaurantId)]

# Obtener id de usuario ordenado por actividad (más reviews)
usr_list = ALL.groupby("userId").like.count().sort_values(ascending=False).reset_index().rename(columns={"like": "new_id_user"})
usr_list["new_id_user"] = list(range(len(usr_list)))
ALL = ALL.merge(usr_list).drop(columns=["userId"]).rename(columns={"new_id_user": "userId"})

# Obtener, para cada restaurante, los usuarios del x%
usr_pctg = .25
max_usr_id = int(len(ALL.userId.unique())*usr_pctg)

RST_USRS = ALL.groupby("restaurantId").apply(lambda x: pd.Series({"rst_name":x.rest_name.unique()[0],"users":x.userId.values[x.userId.values<max_usr_id]})).reset_index()
RST_USRS["n_usrs"] = RST_USRS["users"].apply(len)

def x_t(x):
    s = np.zeros(max_usr_id, dtype=int)
    s[x]=1
    return s

RST_USRS["users"] = RST_USRS["users"].apply(x_t)
RST_USRS = RST_USRS.sort_values("n_usrs", ascending=False)
RST_USRS["id_rest"] = range(len(RST_USRS))

distances = squareform(pdist(np.row_stack(RST_USRS.users.values), metric=np.dot))

'''
CSV = pd.DataFrame(RST_USRS.users.tolist())
CSV["rest_name"] = RST_USRS["rst_name"].apply(lambda x: x.encode("ascii", "ignore").decode())
CSV.to_csv("rst_user_data.csv",encoding='utf-8', index=False)
'''

for _,r in RST_USRS.iterrows():
    idr = r.id_rest
    closest = np.argsort(-distances[idr,:])[:5]
    closest = RST_USRS.loc[RST_USRS.id_rest.isin(closest), "rst_name"].tolist()
    print(r.rst_name, list(zip(closest,-np.sort(-distances[idr,:])[:5])))


'''
# Obtener el porcentaje de usuarios requerido
self.DATASET.DATA["N_USR_PCTG"] = int(self.DATASET.DATA["N_USR"] * self.CONFIG["data_filter"]["pctg_usrs"])
print_g("\t· Obtaining the %d%% of the most active users (%d)..." % (int(self.CONFIG["data_filter"]["pctg_usrs"] * 100), self.DATASET.DATA["N_USR_PCTG"]))
'''
