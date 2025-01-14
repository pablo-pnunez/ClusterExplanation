import argparse
import nvgpu

from src.datasets.semantics.OnlyFoodAndImages import *
from src.datasets.semantics.OnlyFoodAndImagesIntersection import *

from src.models.recomendation.Multilabel import *
from src.models.recomendation.MultilabelSmp import *
from src.models.recomendation.Multiclass import *
from src.models.recomendation.MulticlassSmp import *

########################################################################################################################

def cmdArgs():
    # Obtener argumentos por linea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, help="Test index")
    parser.add_argument('-c', type=str, help="City", )
    parser.add_argument('-s', type=str, help="Stage", )
    parser.add_argument('-gpu', type=str, help="Gpu")
    parser.add_argument('-seed', type=int, help="Seed")
    parser.add_argument('--log2file', dest='log2file', action='store_true')
    args = parser.parse_args()
    return args

########################################################################################################################

# DESCARTADO; NO SE PUEDE GANAR AL RECOMENDADOR DE TRIPADVISOR [Baseline popularidad] (NUESTROS DATOS ESTÁN CONDICIONADOS POR ESTE)

args = cmdArgs()

stage = "baseline" if args.s is None else args.s

seed = 100 if args.seed is None else args.seed
gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"],nvgpu.gpu_info())))) if args.gpu is None else args.gpu

city  = "london" if args.c is None else args.c
city = city.lower().replace(" ","")

position_mode="max"

# DATASETS #############################################################################################################

#data_cfg  = {"city":city,"data_path":"/media/HDD/pperez/TripAdvisor/"+city+"_data/"}
#dts = OnlyFoodAndImages(data_cfg)

data_cfg  = {"city":city,"base_data_path":"/media/HDD/pperez/TripAdvisor/","data_path":"/media/HDD/pperez/TripAdvisor/"+str(city)+"_data/", "cities":["gijon","barcelona","madrid","paris","newyorkcity","london"]}
dts = OnlyFoodAndImagesIntersection(data_cfg)

# MODELS ###############################################################################################################

cfg_u = {"model":{"learning_rate":1e-3, "epochs":150, "batch_size":1024,"seed":seed}, "session":{"gpu":gpu}}
#mdl = Multilabel(cfg_u, dts) # 600/400
#mdl = MultilabelSmp(cfg_u, dts) # 256/512
#mdl = Multiclass(cfg_u, dts) # 600/400
mdl = MulticlassSmp(cfg_u, dts) # 256/512

# STAGES ###############################################################################################################

if "train" in stage: mdl.train(save=True, log2file=args.log2file)
if "test"  in stage: mdl.test(log2file=args.log2file,position_mode=position_mode)
if "baseline"  in stage: mdl.test_baseline(position_mode=position_mode)
