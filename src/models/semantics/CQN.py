# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd

from src.models.semantics.ModelSemantics import *
from src.Common import print_e

import keras
import keras.backend as K
from scipy.spatial.distance import cdist

from keras.layers import Flatten, Embedding, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MultiLabelBinarizer

########################################################################################################################

class CQN(ModelSemantics):

    def __init__(self,config, dataset):
        ModelSemantics.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        def getSharedModel(input_shape):

            def cerquinasFn(x):
                usr = x[0]
                img = x[1]

                n_usr = K.pow(tf.norm(usr, axis=1), 2)
                n_img = K.pow(tf.norm(img, axis=1), 2)

                dot = K.batch_dot(usr, img, axes=1)[:, 0]

                return -n_usr - n_img + (2 * dot)

            emb_size = 256

            input_i = keras.layers.Input(shape=input_shape)
            # model_i = Dropout(.5)(input_i)
            model_i = Dense(512, activation='relu')(input_i)
            # model_i = Dropout(.5)(input_i)
            model_i = Dense(emb_size, name="img_emb")(model_i)
            # model_i = Dense(emb_size, name="img_emb",kernel_constraint=keras.constraints.UnitNorm(axis=0))(model_i)

            input_u = keras.layers.Input(shape=(1,))

            model_u = Embedding(self.DATASET.DATA["N_USR"], emb_size)(input_u)
            model_u = Flatten()(model_u)

            cerquinas = keras.layers.Lambda(cerquinasFn)([model_u, model_i])
            # cerquinas = keras.layers.Dot(axes=1)([model_u, model_i])

            return Model(inputs=[input_u, input_i], outputs=[cerquinas])

        def max_margin(good, bad):
            def loss(y_true, y_pred):
                return K.sum(K.maximum(tf.constant(0, dtype=tf.float32), 1 - good + bad))
                # return K.reduce_sum(K.maximum(tf.constant(0, dtype=tf.float32), 1 - good + bad))

            return loss

            # Fijar las semillas de numpy y TF

        sharedModel = getSharedModel((self.DATASET.DATA["V_IMG"],))

        in_usr = keras.layers.Input(shape=(1,))
        in_good = keras.layers.Input(shape=(self.DATASET.DATA["V_IMG"],))
        in_bad = keras.layers.Input(shape=(self.DATASET.DATA["V_IMG"],))

        shared_good = sharedModel([in_usr, in_good])
        shared_bad = sharedModel([in_usr, in_bad])

        opt = Adam(lr=self.CONFIG["learning_rate"])

        model_take = Model(inputs=[in_usr, in_good, in_bad], outputs=[shared_good, shared_bad])

        model_take.compile(optimizer=opt, loss=max_margin(good=shared_good, bad=shared_bad))

        return model_take

    def train(self, save=False, continue_from=0):

        if os.path.exists(self.MODEL_PATH) and continue_from==0:
            print_e("The model already exists...")
            return
        else:
            if save: os.makedirs(self.MODEL_PATH, exist_ok=True)

        # Conjuntos de entrenamiento
        train_sequence = self.Sequence(self)

        callbacks = []

        if os.path.exists(self.LOG_PATH) and continue_from == 0:
            self.printE("TensorBoard path already exists...")
            exit()

        dev_call = self.DevCallback(self)
        callbacks.append(dev_call)

        if save:
            os.makedirs(self.LOG_PATH, exist_ok=True)
            tb_call = keras.callbacks.TensorBoard(log_dir=self.LOG_PATH, update_freq='epoch')
            mc = ModelCheckpoint(self.MODEL_PATH+"/weights", save_weights_only=True, save_best_only=True, monitor="model_b_loss")

            callbacks.append(tb_call)
            callbacks.append(mc)


        if continue_from > 0:
            self.MODEL.load_weights(self.MODEL_PATH + "/weights")

        self.MODEL.fit_generator(train_sequence,
                                 steps_per_epoch=train_sequence.__len__(),
                                 epochs=self.CONFIG["epochs"],
                                 verbose=1,
                                 workers=1,
                                 callbacks=callbacks,
                                 max_queue_size=25,
                                 initial_epoch=continue_from - 1 if continue_from > 0 else continue_from)

        K.clear_session()

    def __get_image_encoding__(self, encoding="emb"):

        if "emb" in encoding:
            self.MODEL.load_weights(self.MODEL_PATH + "/weights")
            sub_model = Model(inputs=[self.MODEL.get_layer("model_1").get_layer("input_1").input],outputs=[self.MODEL.get_layer("model_1").get_layer("img_emb").output])
            all_img_embs = sub_model.predict(self.DATASET.DATA["IMG_VEC"], batch_size=self.CONFIG["batch_size"])
        if "densenet" in encoding:
            all_img_embs = self.DATASET.DATA["IMG_VEC"]

        return all_img_embs

    ####################################################################################################################

    class Sequence(Sequence):

        def __init__(self,  model):

            self.MODEL = model
            self.TRAIN_IMGS = np.asarray(self.MODEL.DATASET.DATA["IMG"].loc[self.MODEL.DATASET.DATA["IMG"].test == False].id_img)
            self.init_data()

        def init_data(self):

            def create_triplet_for_user(id_u, rows):

                usrs = np.repeat(id_u, self.MODEL.CONFIG["n_items"] * len(rows))
                usr_rsts = rows.id_restaurant.unique()

                '''
                # CON ADYACENTES ---------------------------------------------------------------------------------------

                if (self.ADY_NEG):

                    good = []
                    bad = []

                    for x in usr_rsts:
                        t_g = np.random.choice(self.MODEL.DATA["TRAIN_RST_IMG"].iloc[x], self.N_ITEMS, replace=True)

                        b_rs = self.MODEL.DATA["RST_ADY"].iloc[x]["ady"]
                        b_r = list(set(b_rs) - set(usr_rsts))

                        good.extend(t_g)

                        # Si no hay adyacentes
                        if (len(b_r) > 0):
                            # b_r = np.random.choice(b_r,1)[0]
                            # t_b = np.random.choice(self.MODEL.DATA["TRAIN_RST_IMG"].iloc[b_r], self.N_ITEMS, replace=True)
                            t_b = np.random.choice(np.concatenate(self.MODEL.DATA["TRAIN_RST_IMG"].iloc[b_r].values), self.N_ITEMS, replace=True)
                            bad.extend(t_b)
                        else:
                            t_b = np.random.choice(self.TRAIN_IMGS, self.N_ITEMS, replace=True)
                            bad.extend(t_b)

                # CON ADYACENTES 2 -------------------------------------------------------------------------------------
                if (self.ADY_NEG):

                    good = []
                    bad = []

                    for x in usr_rsts:
                        t_g = np.random.choice(self.MODEL.DATA["TRAIN_RST_IMG"].iloc[x], self.N_ITEMS, replace=True)

                        b_rs = self.MODEL.DATA["RST_ADY"].iloc[x]["ady"]
                        b_r = list(set(b_rs) - set(usr_rsts))

                        good.extend(t_g)

                        # Si no hay adyacentes
                        if (len(b_r) > 0):
                            b_r = np.random.choice(b_r,1)[0]
                            t_b = np.random.choice(self.MODEL.DATA["TRAIN_RST_IMG"].iloc[b_r], self.N_ITEMS//2, replace=True)
                            bad.extend(t_b)

                            t_b = np.random.choice(self.TRAIN_IMGS, self.N_ITEMS//2, replace=True)
                            bad.extend(t_b)

                        else:
                            t_b = np.random.choice(self.TRAIN_IMGS, self.N_ITEMS, replace=True)
                            bad.extend(t_b)
                '''
                # ALEATORIO --------------------------------------------------------------------------------------------

                good = np.asarray(
                    [np.random.choice(self.MODEL.DATASET.DATA["TRAIN_RST_IMG"].iloc[x], self.MODEL.CONFIG["n_items"], replace=True) for x in
                     usr_rsts]).flatten()
                bad = np.random.choice(self.TRAIN_IMGS, self.MODEL.CONFIG["n_items"] * len(rows), replace=True)

                # good = np.asarray([random.choices(self.MODEL.DATA["TRAIN_RST_IMG"].iloc[x], k=self.N_ITEMS) for x in usr_rsts]).flatten()
                # bad = random.choices(self.TRAIN_IMGS, k=self.N_ITEMS * len(rows))

                # ------------------------------------------------------------------------------------------------------

                return list(zip(usrs, good, bad))

            ############################################################################################################

            # Generar el conjunto para esta epoch

            ret = []
            ret_cols = ["id_user", "good", "bad"]

            for id_u, rows in tqdm(self.MODEL.DATASET.DATA["TRAIN"].groupby("id_user")):
                usr_triplets = create_triplet_for_user(id_u, rows)
                ret.extend(usr_triplets)

            ret = pd.DataFrame(ret, columns=ret_cols).sample(frac=1)

            # Generar los batches para esta epoch

            if (len(ret) > self.MODEL.CONFIG["batch_size"]):
                self.BATCHES = np.array_split(ret, len(ret) // self.MODEL.CONFIG["batch_size"])
            else:
                self.BATCHES = np.array_split(ret, 1)

        def __len__(self):
            return len(self.BATCHES)

        def __getitem__(self, idx):

            # print("\nPIDIENDO ITEM %d" % idx)

            batch = self.BATCHES[idx]

            good = self.MODEL.DATASET.DATA["IMG_VEC"][batch.good.astype(int).values]
            bad = self.MODEL.DATASET.DATA["IMG_VEC"][batch.bad.astype(int).values]

            return ([np.asarray(batch.id_user), good, bad], [np.zeros(len(batch)), np.zeros(len(batch))])

        def on_epoch_end(self):

            self.init_data()

    class DevCallback(tf.keras.callbacks.Callback):

        def __init__(self, parent):
            self.parent = parent

        def on_epoch_end(self, epoch, logs=None):

            logs["model_b_loss"] = logs["model_1_loss"] / self.parent.CONFIG["batch_size"]

