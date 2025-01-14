import argparse
import nvgpu
import time

from src.datasets.semantics.OnlyFood import *
from src.datasets.semantics.OnlyFoodAndImages import *
from src.datasets.semantics.OnlyFoodAndImagesT2 import *

from src.models.semantics.SemPic import *
from src.models.semantics.SemPic2 import *
from src.models.semantics.SemPic2D import *

########################################################################################################################

def cmdArgs():
    # Obtener argumentos por linea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('-pctU', type=float, help="User percentage")
    parser.add_argument('-activeU', type=int, help="Active users or random")
    parser.add_argument('-seed', type=int, help="Seed")
    #parser.add_argument('-tstmp', type=int, help="Timestamp")
    parser.add_argument('-c', type=str, help="City", )
    parser.add_argument('-s', type=str, help="Stage", )
    parser.add_argument('-e', type=str, help="Embedding", )
    parser.add_argument('-gpu', type=str, help="Gpu")
    parser.add_argument('--log2file', dest='log2file', action='store_true')
    args = parser.parse_args()
    return args

########################################################################################################################

args = cmdArgs()

stage = "test2" if args.s is None else args.s
encoding = "emb" if args.e is None else args.e

#tstmp = int(time.time()*1000) if args.tstmp is None else args.tstmp
pctg_usrs = .05 if args.pctU is None else args.pctU
active_usrs = 0 if args.activeU is None else args.activeU
seed = 100 if args.seed is None else args.seed
gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"],nvgpu.gpu_info())))) if args.gpu is None else args.gpu

city  = "gijon" if args.c is None else args.c
city = city.lower().replace(" ","")

# DATASETS #############################################################################################################

data_cfg  = {"city":city,"data_path":"/media/HDD/pperez/TripAdvisor/"+city+"_data/"}
#dts = OnlyFood(data_cfg)
dts = OnlyFoodAndImages(data_cfg)

#exit()

#data_cfg  = {"city":city,"base_data_path":"/media/HDD/pperez/TripAdvisor/","data_path":"/media/HDD/pperez/TripAdvisor/"+str(city)+"_data/", "cities":["gijon","barcelona","madrid","paris","newyorkcity","london"]}
#dts = OnlyFoodAndImagesT2(data_cfg) # Con los 2 TEST (uno con usuarios de la ciudad y otro con los que fueron a ciudad y otras)

#dts.dataset_stats()
#dts.test_baseline_hdp()
#dts.rest_most_popular()

exit()
# MODELS ###############################################################################################################

# RecSys 2020 ----------------------------------------------------------------------------------------------------------

#seeds = [100,12,8778,0,99968547,772,8002,4658,9,34785]
#cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":5e-4, "epochs":500, "batch_size":2048, "gpu":gpu,"seed":seeds[cfg_no]}
#mdl = SemPic(cfg_u, dts)

# RecSys 2020 ++ -------------------------------------------------------------------------------------------------------

#seeds = [100,12,8778,0,99968547,772,8002,4658,9,34785]
#pctg_usrs = [.05,.1,.15,.2,.25,.3,.35]

cfg_u = {"model":{"pctg_usrs":pctg_usrs, "active_usrs":bool(active_usrs), "learning_rate":1e-3, "epochs":500, "batch_size":1024,"seed":seed}, "session":{"gpu":gpu}}
mdl = SemPic2(cfg_u, dts)

#ToDo: Densenet

# GridSearch -----------------------------------------------------------------------------------------------------------

#cfg_no = 0;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":5e-4, "epochs":500, "batch_size":2048, "gpu":gpu,"seed":100}
#cfg_no = 1;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":1e-4, "epochs":500, "batch_size":2048, "gpu":gpu,"seed":100}
#cfg_no = 2;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":1e-3, "epochs":500, "batch_size":2048, "gpu":gpu,"seed":100}

##cfg_no = 3;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":1e-3, "epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}
#cfg_no = 4;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":1e-3, "epochs":500, "batch_size":4096, "gpu":gpu,"seed":100}

#cfg_no = 5;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":1e-3, "epochs":500, "batch_size":512, "gpu":gpu,"seed":100}
#cfg_no = 6;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":1e-3, "lr_decay":True, "epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}

#cfg_no = 7;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":5e-4, "lr_decay":True, "epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}
#cfg_no = 8;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":1e-4, "lr_decay":True, "epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}

#cfg_no = 9;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "active_usrs":True, "learning_rate":1e-3,"epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}
#cfg_no = 10;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "active_usrs":True, "learning_rate":5e-4,"epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}
#cfg_no = 11;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "active_usrs":True, "learning_rate":1e-4,"epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}

#cfg_no = 12;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.50, "active_usrs":True, "learning_rate":1e-3,"epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}
#cfg_no = 13;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.75, "active_usrs":True, "learning_rate":1e-3,"epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}

#cfg_no = 14;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.10, "active_usrs":False, "learning_rate":1e-3,"epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}
#cfg_no = 15;cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.10, "active_usrs":True, "learning_rate":1e-3,"epochs":500, "batch_size":1024, "gpu":gpu,"seed":100}

#mdl = SemPic2(cfg_u, dts)
#mdl = SemPic2D(cfg_u, dts)

# STAGES ###############################################################################################################

if "train" == stage: mdl.train(save=True, log2file=args.log2file)

if "test" == stage: mdl.test(encoding=encoding, log2file=args.log2file)
if "test2" == stage: mdl.test2(encoding=encoding, log2file=args.log2file) # Test utilizando el resto de ciudades para evaluar, creo....
if "test2bl" == stage: mdl.test2(encoding=encoding, baseline=True, log2file=args.log2file)


if "plot" == stage: mdl.emb_tsne(layer_name="img_emb")
if "othr" == stage: mdl.find_image_semanthics()
