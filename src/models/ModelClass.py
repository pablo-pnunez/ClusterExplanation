# -*- coding: utf-8 -*-
from src.Common import print_w

import random
import os

import json
import hashlib

import numpy as np
import tensorflow as tf

class ModelClass:

    def __init__(self,config,dataset):
        self.CONFIG = config
        self.DATASET = dataset

        self.MODEL_NAME = self.__class__.__name__
        # self.CUSTOM_PATH = self.MODEL_NAME+"/"+self.CONFIG["id"]+"/"

        # El MD5 sin el gpu para que no cambie
        self.MD5, cfg_save_data = self.__get_md5__()
        self.CUSTOM_PATH = self.MODEL_NAME+"/"+self.DATASET.CONFIG["city"]+"/"+self.MD5+"/"

        self.MODEL_PATH = "models/"+self.CUSTOM_PATH
        self.LOG_PATH = "logs/"+self.CUSTOM_PATH

        # Crear carpeta para el modelo
        if os.path.exists(self.MODEL_PATH): print_w("The model already exists...");
        else: os.makedirs(self.MODEL_PATH, exist_ok=True)

        # Crear json con la configuración del modelo
        #with open(self.MODEL_PATH+'/cfg.json', 'w') as fp: json.dump(self.CONFIG["model"], fp)
        with open(self.MODEL_PATH+'/cfg.json', 'w') as fp: json.dump(cfg_save_data, fp,indent=4)

        # Fijar las semillas de numpy y TF
        np.random.seed(self.CONFIG["model"]["seed"])
        random.seed(self.CONFIG["model"]["seed"])
        tf.random.set_seed(self.CONFIG["model"]["seed"])

        # Seleccionar la GPU más adecuada y limitar el uso de memoria
        self.__config_session__()

        #Crear el modelo
        self.MODEL = self.get_model()

    def __get_md5__(self):
        # Obtiene las partes del dicionario de configuración que pueden alterar un modelo (i.e la gpu no, pero el porcentaje de usuarios si). Retorna el MD5
        res = {"dataset_config": self.DATASET.CONFIG}
        for k in self.CONFIG.keys():
            if ("in_md5" not in self.CONFIG[k].keys()) or ("in_md5" in self.CONFIG[k].keys() and self.CONFIG[k]["in_md5"]) :
                res[k] = self.CONFIG[k]
        return hashlib.md5(str(res).encode()).hexdigest(), res

    def __config_session__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.CONFIG["session"]["gpu"])

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)

    def get_model(self):
        raise NotImplementedError

    def train(self, save=False):
        raise NotImplementedError

