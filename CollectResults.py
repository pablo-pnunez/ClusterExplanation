import os, shutil
import json
import pandas as pd
import numpy as np

########################################################################################################################

def read_semantics(test_path, cfg):

    # Leer los datos de test
    with open(test_path) as test_file:
        lines = test_file.readlines()
        if len(lines) < 12:
            print(test_path)
            return []
        one = float(lines[-7].strip())
        two = float(lines[-5].strip())
        three = float(lines[-3].strip())
        four = float(lines[-1].strip())

        line = [city]
        line.extend(cfg.values())
        line.extend((one, two, three, four))

    return line

def read_recomendations(test_path, cfg):
    # Leer los datos de test
    with open(test_path) as test_file:
        lines = test_file.readlines()
        if len(lines) < 7:
            print(test_path)
            return []

        d1 = list(map(lambda x: float(x),lines[-4].split("\t")))
        d2 = list(map(lambda x: float(x),lines[-3].split("\t")))
        d3 = list(map(lambda x: float(x),lines[-2].split("\t")))
        d4 = list(map(lambda x: float(x),lines[-1].split("\t")))

        line = [city]
        line.extend(cfg.values())
        line.extend(d1)
        line.extend(d2)
        line.extend(d3)
        line.extend(d4)

    return line

########################################################################################################################

collect_results("barcelona")

exit()

cities = ["gijon","barcelona","madrid","paris","newyorkcity","london"]
#cities = ["london"]
model = "SemPic2"#"MulticlassSmp"#"Multilabel"#"SemPic2"
ret = []
test_mode="2"#"_"+"max"

incomplete = []

if "SemPic2" in model:
    last_columns = ("1","2","3","4")
    read_test = read_semantics

if ("Multilabel" in model) or ("Multiclass" in model):
    last_columns = ["n1","ni1","mdn1","avg1","std1",
                    "n2","ni2","mdn2","avg2","std2",
                    "n3","ni3","mdn3","avg3","std3",
                    "n4","ni4","mdn4","avg4","std4"]
    read_test = read_recomendations

for city in cities:
    base_path = "/home/pperez/PycharmProjects/SemPic/models/"+model+"/"+city+"/"

    for f in os.listdir(base_path):
        folder_path =  base_path + f +"/"
        test_path = folder_path+"test"+test_mode+".txt"
        train_path = folder_path+"train.txt"
        cfg_path  = folder_path+"cfg.json"

        #os.rename(test_path.replace(test_mode,""),test_path);continue
        #os.remove(test_path);continue

        # Verificar si se hicieron 500 epochs, si no se eliminan
        with open(train_path) as train_file:
            t_lines = train_file.readlines()
            try:
                t_last = list(map(lambda x: int(x), t_lines[-2].strip().split(" ")[1].split("/")))
                if (np.diff(t_last)[0] != 0): incomplete.append(folder_path)
            except:
                incomplete.append(folder_path)


        #Si hay test, leer
        if os.path.exists(test_path):

            with open(cfg_path) as json_file: cfg = json.load(json_file)

            line = read_test(test_path, cfg)
            ret.append(line)

columns = ["city"]
columns.extend(cfg.keys())
columns.extend(last_columns)

ret = pd.DataFrame(ret, columns = columns)
#ret = ret.sort_values("pctg_usrs", ascending=False).reset_index(drop=True)


for rem in incomplete:
    print(rem)
    #shutil.rmtree(rem)


if "SemPic2" in model:

    for city, c_d in ret.groupby("city"):
        print(city)
        for active, c_a in c_d.groupby("active_usrs"):
            print(active)
            for pct, c_p in c_a.groupby("pctg_usrs"):
                if len(c_p)==10 :
                    print("%f\t%f\t%f\t%f\t%f\t\t%f\t%f\t%f\t%f" % (pct,c_p["1"].mean(),c_p["2"].mean(),c_p["3"].mean(),c_p["4"].mean(),c_p["1"].std(),c_p["2"].std(),c_p["3"].std(),c_p["4"].std()))

if ("Multilabel" in model) or ("Multiclass" in model):
    for city, c_p in ret.groupby("city"):
        print(city)
        if len(c_p) == 10:
            print("%f\t%f\t%f\t%f\t\t%f\t%f\t%f\t%f\t\t%f\t%f\t%f\t%f" % (c_p["mdn1"].mean(),c_p["mdn2"].mean(),c_p["mdn3"].mean(), c_p["mdn4"].mean(),
                                                                            c_p["avg1"].mean(),c_p["avg2"].mean(),c_p["avg3"].mean(), c_p["avg4"].mean(),
                                                                            c_p["std1"].mean(),c_p["std2"].mean(),c_p["std3"].mean(), c_p["std4"].mean()))