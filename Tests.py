import nvgpu
import numpy as np

from src.Common import to_pickle, get_pickle, print_g
from src.datasets.semantics.OnlyFoodAndImages import *
from src.models.semantics.SemPic2 import *
from scipy.spatial.distance import cdist, pdist, squareform

# ----------------------------------------------------------------------------------------------------------------------

city  = "gijon"
pctg_usrs = .15
active_usrs = 1
seed = 100
gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"],nvgpu.gpu_info()))))

data_cfg  = {"city":city,"data_path":"/media/HDD/pperez/TripAdvisor/"+city+"_data/"}
dts = OnlyFoodAndImages(data_cfg)


# ----------------------------------------------------------------------------------------------------------------------
# PREDICCIÓN DE NUESTRO MODELO
# ----------------------------------------------------------------------------------------------------------------------

cfg_u = {"model":{"pctg_usrs":pctg_usrs, "active_usrs":bool(active_usrs), "learning_rate":1e-3, "epochs":500, "batch_size":1024,"seed":seed}, "session":{"gpu":gpu}}
mdl = SemPic2(cfg_u, dts)


model = mdl.MODEL
model.load_weights(mdl.MODEL_PATH + "weights")

sub_model = Model(inputs=[model.get_layer("in").input], outputs=[model.get_layer("img_emb").output])

output_embs = model.predict(dts.DATA["IMG_VEC"])
sempic_embs = sub_model.predict(dts.DATA["IMG_VEC"])

output_embs = output_embs[:10,:]
sempic_embs = sempic_embs[:10,:]

dist_o = pdist(output_embs, lambda u, v: np.dot(u,v))
dist_s = pdist(sempic_embs, lambda u, v: np.dot(u,v))

print(np.corrcoef(dist_o, dist_s)[0,1])

