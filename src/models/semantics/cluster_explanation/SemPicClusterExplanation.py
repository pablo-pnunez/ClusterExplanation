# -*- coding: utf-8 -*-
import shutil

from src.models.semantics.SemPic import *
from scipy.spatial.distance import cdist

from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow.python.ops import clip_ops, math_ops
import tensorflow_addons as tfa

import matplotlib.pylab as plt
import seaborn as sns
import tensorflow as tf
import cv2

class SemPicClusterExplanation(SemPic):

    def __init__(self,config, rest_cluster, dataset):
        self.REST_CLUSTER = rest_cluster
        SemPic.__init__(self, config=config, dataset=dataset)
        self.TRAIN_SEQUENCE = None
        self.DEV_SEQUENCE = None

    def get_model(self):

        the_model = self.CONFIG["model"]["model_version"]

        if the_model == "a":
            return self.get_model_a();
            
        elif the_model == "h":
            return self.get_model_h();  # -> El mejor (0.165)

        elif "hl" in the_model:
            class_one_weight = int(the_model.replace("hl",""))
            return self.get_model_hl(class_one_weight);

    def get_model_a(self):

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = input
        x = Dropout(.6)(x)
        output = Dense(self.REST_CLUSTER.DATA["N_USR_PCTG"], activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=opt, loss= tf.keras.losses.BinaryCrossentropy(), metrics=[self.precision, self.recall, self.f1, "accuracy"])

        return model

    def get_model_h(self):

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = input
        x = Dense(600, activation="relu")(x)
        x = Dropout(.6)(x)
        x = BatchNormalization()(x)
        x = Dense(400, activation="relu")(x)
        x = Dropout(.3)(x)
        output = Dense(self.REST_CLUSTER.DATA["N_USR_PCTG"], activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=opt, loss= tf.keras.losses.BinaryCrossentropy(), metrics=[self.precision, self.recall, self.f1, "accuracy"])

        return model

    def get_model_hl(self, class_one_weight):

        def create_weighted_binary_crossentropy(zero_weight, one_weight):
            def weighted_binary_crossentropy(y_true, y_pred):
                y_true = K.cast(y_true, dtype=tf.float32)

                # Calculate the binary crossentropy
                b_ce = K.binary_crossentropy(y_true, y_pred)

                # Apply the weights
                weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
                weighted_b_ce = weight_vector * b_ce

                # Return the mean error
                return K.mean(weighted_b_ce)

            return weighted_binary_crossentropy

        print_w("Class weights: [%d-%d]" % (1,class_one_weight))

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = input
        x = Dense(600, activation="relu")(x)
        x = Dropout(.6)(x)
        x = BatchNormalization()(x)
        x = Dense(400, activation="relu")(x)
        x = Dropout(.3)(x)
        output = Dense(self.REST_CLUSTER.DATA["N_USR_PCTG"], activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=opt, loss= create_weighted_binary_crossentropy(1,class_one_weight), metrics=[self.precision, self.recall, self.f1, "accuracy"])
        # model.compile(optimizer=opt, loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=[self.precision, self.recall, self.f1, "accuracy"])

        return model

    ####################################################################################################################

    def dev_stats(self):

        # Cargar pesos
        self.MODEL.load_weights(self.MODEL_PATH + "weights")
        # Obtener predicciones
        pred = self.MODEL.predict(self.DEV_SEQUENCE)
        real = self.DEV_SEQUENCE.KHOT.fit_transform(self.DEV_SEQUENCE.ALL_DATA.output)
        # Distribución predicha
        dist_pred = pred.sum(0)
        # Distribución predicha discretizada
        dist_pred_disc = ((pred>.5)*1).sum(0)
        # Distribución original
        dist_real = real.sum(0)
        # Dataframe
        all_data = pd.DataFrame(zip(np.tile(range(len(dist_real)),3), np.concatenate([["real"]*len(dist_real), ["prediccion (prob)"]*len(dist_real),["prediccion (0/1)"]*len(dist_real)]), np.concatenate([dist_real,dist_pred,dist_pred_disc])),
                                columns=["usr","type","count"]).sort_values(["usr", "type"])

        (_, k_prec, k_rec, k_f1, _) = self.MODEL.evaluate(self.DEV_SEQUENCE, verbose=0)

        thresholds = np.asarray(list(range(1, 10)))/10
        th_res = []
        for thres in thresholds:
            s_prec = precision_score(real, (pred > thres) * 1, average='micro')
            s_rec = recall_score(real, (pred > thres) * 1, average='micro')
            s_f1 = f1_score(real, (pred > thres) * 1, average='micro')

            th_res.extend(zip([thres]*3, [s_prec, s_rec, s_f1],["precision","recall","f1"]))

        th_res = pd.DataFrame(th_res, columns=["thres", "value", "metric"])
        the_plot = sns.barplot(data=th_res, x="thres", y="value", hue="metric")
        the_plot.set_title("Métricas variando el umbral para %s" % ( self.DATASET.CONFIG["city"]))
        the_plot.set_xlabel("Umbral")
        the_plot.set_ylabel("Valor")
        the_plot.grid()
        the_plot.axhline(k_f1, ls="--", color="red")
        plt.show()

        # Plot
        n_usrs_show = 25
        all_data_plot = all_data.iloc[:n_usrs_show * 3]

        plt.figure(figsize=(12,6))
        the_plot = sns.barplot(data=all_data_plot, x="usr", y="count", hue="type")
        the_plot.grid()
        the_plot.set_title("Número de unos para los %d primeros usuarios sensores en DEV [%s]" % (n_usrs_show, self.DATASET.CONFIG["city"]))
        plt.show()

        '''
        # Diferencia entre número de unos real y predicho
        user_difference = all_data.groupby("usr").apply(lambda x: pd.Series({"diff": ((x.loc[x.type=="real"]["count"].values[0] - x.loc[x.type=="prediccion (0/1)"]["count"].values[0])/x.loc[x.type=="real"]["count"].values[0])*100})).reset_index()
        print(user_difference["diff"].describe())
        # user_difference_plot = user_difference.iloc[:n_usrs_show]
        steps = np.arange(user_difference["diff"].min(), user_difference["diff"].quantile(.9))
        plt.figure(figsize=(10,8))
        the_plot = sns.distplot(user_difference["diff"], bins = steps, kde=False)
        the_plot.grid()
        the_plot.set_title("Distribución de diferencias [real - pred(0/1)] de los usuarios sensores en DEV [%s]\n MEDIA: %.2f STD: %.2f" % ( self.DATASET.CONFIG["city"], user_difference["diff"].mean(), user_difference["diff"].std()))
        the_plot.axvline(user_difference["diff"].mean(), ls="--")
        the_plot.set_xlim([user_difference["diff"].min(), user_difference["diff"].quantile(.9)])
        the_plot.set_xticks(steps)
        the_plot.set_xticklabels(steps, rotation=45, fontdict={'fontsize': 8})
        plt.show()
        '''

    def train(self, save_model=True):

        # Si no existe la carpeta dev, es que está aqui lo de dev, hay que moverlo
        if not os.path.exists(self.MODEL_PATH+"dev/"):
            os.makedirs(self.MODEL_PATH+"dev/")
            for file_name in os.listdir(self.MODEL_PATH):
                shutil.move(os.path.join(self.MODEL_PATH, file_name), self.MODEL_PATH+"dev/")

        # Entrenar normal sin separar imágenes para DEV
        self.TRAIN_SEQUENCE = self.Sequence_DEV(self, dev_images=[])
        self._train_model(save_model=save_model, final=True)

    def train_dev(self, save_model=False):

        def _get_dev_imgs(x):
            n_items = int(sum(x.num_images.values)*dev_size)
            imgs_ids = self.DATASET.DATA["IMG"].loc[self.DATASET.DATA["IMG"].reviewId.isin(x.reviewId.values)].sample(n_items, random_state=self.CONFIG["model"]["seed"])
            return pd.Series({"imgs_ids":imgs_ids.id_img.values, "n_items":n_items})

        # ---------

        # Dev size (Porcentaje de fotos de cada restaurante que se usan para DEV)
        dev_size = .15
        # Se buscan las reviews con imagen y, para cada restaurante, nos quedamos con el X % de sus imágenes para DEV
        valid_reviews = self.DATASET.DATA["TRAIN"].loc[self.DATASET.DATA["TRAIN"].num_images>0].groupby("id_restaurant").apply(lambda x: _get_dev_imgs(x)).reset_index()
        # Obtenemos la lista de esas imágenes
        dev_images = np.concatenate(valid_reviews.loc[valid_reviews.n_items>0].imgs_ids.values)
        print_w("%d dev images." % len(dev_images))

        # La Sequence de Train eliminará estos ejemplos, y la Sequence de test solo pondrá estos ejemplos

        self.TRAIN_SEQUENCE = self.Sequence_DEV(self, dev_images=dev_images)
        self.DEV_SEQUENCE = self.Sequence_DEV(self, dev = True, dev_images=dev_images)

        zrs_ons = self.TRAIN_SEQUENCE.ALL_DATA.output.apply(lambda x: pd.Series({"ones": len(x), "zeros": self.REST_CLUSTER.DATA["N_USR_PCTG"] - len(x)}))
        zrs_ons_total = zrs_ons.sum().values
        print_g("Hay %d unos y %d ceros. %.2f veces más ceros que unos" % (zrs_ons_total[0], zrs_ons_total[1], zrs_ons_total[1]/zrs_ons_total[0]))
        print_g("De media, para cada ejemplo, hay %.2f veces más ceros que unos" % ((zrs_ons.zeros/zrs_ons.ones).mean()))

        self._train_model(save_model=save_model)

        exit()

    def _train_model(self, save_model=False, final=False):

        def linear_decay(current_epoch, current_lr, initial_rl, final_lr, epochs):
            step = (initial_rl - (1e-5)) / epochs

            return current_lr - step

        def cosine_decay(current_epoch, current_lr, initial_rl, epochs, alpha=0.0):

            step = min(current_epoch, epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * step / epochs))
            decayed = (1 - alpha) * cosine_decay + alpha
            decayed_learning_rate = initial_rl * decayed
            tf.summary.scalar('learning_rate', decayed_learning_rate)

            return decayed_learning_rate

        class CustomStopper(tf.keras.callbacks.EarlyStopping):
            def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, start_epoch=100, restore_best_weights=False):  # add argument for starting epoch
                super(CustomStopper, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)
                self.start_epoch = start_epoch

            def on_epoch_end(self, epoch, logs=None):
                if epoch >= self.start_epoch:
                    super().on_epoch_end(epoch, logs)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        callbacks = []

        lrs = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: linear_decay(epoch, lr, self.CONFIG["model"]["learning_rate"], self.CONFIG["model"]["learning_rate"]/2,
                                                                                     self.CONFIG["model"]['epochs']))
        callbacks.append(lrs)

        # La carpeta cambia si es train_dev o train final
        final_folder = "dev/" if not final else ""

        # El número de epochs es el que se hizo en dev o el que se obtenga utilizando un early stopping
        if final:
            dev_log_path = self.MODEL_PATH+"dev/log.csv"
            if os.path.exists(dev_log_path):
                dev_log_data = pd.read_csv(dev_log_path)
                final_epoch_number = dev_log_data.val_f1.argmax()
            else:
                print_e("Unknown DEV epoch number...")
                exit()
        else:
            est = CustomStopper(monitor="val_f1", start_epoch=100, patience=100, verbose=1, mode="max")
            callbacks.append(est)

        # Si se quiere almacenar la salida del modelo (pesos/csv)
        if save_model:

            if os.path.exists(self.MODEL_PATH + final_folder + "checkpoint"):
                overwrite = input("Model already exists. Do you want to overwrite it? (y/n)")
                if overwrite=="y":
                    sure = input("Are you sure? (y/n)")
                    if sure != "y": return
                else: return

            os.makedirs(self.MODEL_PATH + final_folder, exist_ok=True)
            log = tf.keras.callbacks.CSVLogger(self.MODEL_PATH + final_folder + "log.csv", separator=',', append=False)
            callbacks.append(log)

            # Solo guardar mirando f1 en val cuando hay DEV
            if not final:
                mc = tf.keras.callbacks.ModelCheckpoint(self.MODEL_PATH + final_folder + "weights", save_weights_only=True, save_best_only=True, monitor="val_f1", mode="max")
                callbacks.append(mc)

        else:
            print_g("Not saving the model...")

        # Si es el entrenamiento final, no hay dev
        if final:
            hist = self.MODEL.fit(self.TRAIN_SEQUENCE,
                                  steps_per_epoch=self.TRAIN_SEQUENCE.__len__(),
                                  epochs=final_epoch_number,
                                  verbose=2,
                                  workers=1,
                                  callbacks=callbacks,
                                  max_queue_size=40)

            self.MODEL.save_weights(self.MODEL_PATH + "weights")


        else:
            hist = self.MODEL.fit(self.TRAIN_SEQUENCE,
                                  steps_per_epoch=self.TRAIN_SEQUENCE.__len__(),
                                  epochs=self.CONFIG["model"]['epochs'],
                                  verbose=2,
                                  validation_data=self.DEV_SEQUENCE,
                                  validation_steps=self.DEV_SEQUENCE.__len__(),
                                  workers=1,
                                  callbacks=callbacks,
                                  max_queue_size=40)

        # Almacenar gráfico con el entrenamiento
        if save_model:
            done_epochs = len(hist.history["loss"])
            plt.figure(figsize=(int((done_epochs*8)/500), 8))
            hplt = sns.lineplot(range(done_epochs), hist.history["f1"], label="f1")
            if not final:
                hplt = sns.lineplot(range(done_epochs), hist.history["val_f1"], label="val_f1")
            hplt.set_yticks(np.asarray(range(0, 110, 10)) / 100)
            hplt.set_xticks(range(0, done_epochs, 20))
            hplt.set_xticklabels(range(0, done_epochs, 20), rotation=45)
            hplt.set_title("Train history")
            hplt.set_xlabel("Epochs")
            hplt.set_ylabel("F1")
            hplt.grid(True)
            if final:
                plt.savefig(self.MODEL_PATH + final_folder + "history.jpg")
            else:
                plt.savefig(self.MODEL_PATH + final_folder + "history.jpg")
            plt.clf()

    ####################################################################################################################

    class Sequence_DEV(tf.keras.utils.Sequence):

        def __init__(self, model, dev = False, dev_images=[]):
            self.MODEL = model
            self.IS_DEV = dev
            self.DEV_IMAGES = dev_images
            self.init_data()

        def init_data(self):

            n_usrs = self.MODEL.REST_CLUSTER.DATA["N_USR_PCTG"]
            self.KHOT = MultiLabelBinarizer(classes=list(range(n_usrs)))

            ret = []

            for id_r, rows in tqdm(self.MODEL.DATASET.DATA["TRAIN"].groupby("id_restaurant"), desc="Sequence data"):
                # Obtener el número de imágenes del restaurante
                rst_imgs = self.MODEL.DATASET.DATA["IMG"].loc[self.MODEL.DATASET.DATA["IMG"].reviewId.isin(rows.reviewId)].id_img.values
                # Asegurarse de que siempre tiene imágenes...
                assert len(rst_imgs)>0
                # Obtener usuarios
                rltd_usrs = self.MODEL.REST_CLUSTER.DATA["RST_ENCODING"].loc[self.MODEL.REST_CLUSTER.DATA["RST_ENCODING"].restaurantId==rows.restaurantId.values[0]].users.values[0]
                # Asegurarse de que los usuarios del restaurante están dentro del 25%
                assert rltd_usrs.max()<n_usrs
                # Asegurarse que el número mínimo de usuarios de cada restaurante es el deseado.
                # assert len(rltd_usrs) >= self.MODEL.CONFIG["data_filter"]["min_usrs_per_rest"]
                # Ver que imágenes caen en train y cuales en test
                img_location = np.asarray([int(x in self.DEV_IMAGES) for x in rst_imgs])
                # Si es TEST con los unos.
                if self.IS_DEV: rst_imgs = rst_imgs[np.argwhere(img_location == 1).flatten()]
                # Si es TRAIN nos quedamos con los 0's de img_location
                else: rst_imgs = rst_imgs[np.argwhere(img_location == 0).flatten()]
                # Crear ejemplos
                if len(rst_imgs)<1: continue
                # Crear ejemplos
                r = [id_r] * len(rst_imgs)
                rn = [rows.rest_name.values[0]] * len(rst_imgs)
                x = rst_imgs
                y = [list(rltd_usrs)] * len(rst_imgs) # En la salida, solo aquellos usuarios que son del 25%

                ret.extend(list(zip(r, rn, x, y)))

            ret = pd.DataFrame(ret, columns=["id_restaurant", "rest_name", "id_img", "output"]).sample(frac=1)

            ############################################################################################################

            self.ALL_DATA = ret

            if (len(ret) > self.MODEL.CONFIG["model"]["batch_size"]):
                self.BATCH = np.array_split(ret, len(ret) // self.MODEL.CONFIG["model"]["batch_size"])

            else:
                self.BATCH = np.array_split(ret, 1)

        def read_imgs(self, id_imgs):


            base_path = "/media/nas/pperez/data/TripAdvisor/%s_data/images_lowres/" % (self.MODEL.DATASET.CONFIG["city"])
            imgs_data  = self.MODEL.DATASET.DATA['IMG'].iloc[id_imgs]
            paths = (base_path+imgs_data.reviewId.astype(str)+"/"+imgs_data.image.astype(str)+".jpg").to_list()
            ret = []

            for img_path in paths:
                tmp_img = cv2.imread(img_path)
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                tmp_img = cv2.resize(tmp_img, self.MODEL.CONFIG["model"]["input_shape"][:2])
                ret.append(tmp_img)

            return np.asarray(ret)

        def __len__(self):
            return len(self.BATCH)

        def __getitem__(self, idx):

            btch = self.BATCH[idx]
            y = self.KHOT.fit_transform(btch.output.values)*1.0  # *1 Para que no de error la focal loss

            if "input_shape" in self.MODEL.CONFIG["model"].keys():
                x = tf.keras.applications.densenet.preprocess_input(self.read_imgs(btch.id_img.values))
            else:
                x = self.MODEL.DATASET.DATA['IMG_VEC'][btch.id_img.values]

            return (x, y)

    class Sequence(tf.keras.utils.Sequence):

        def __init__(self, model):
            self.MODEL = model
            self.init_data()

        def init_data(self):

            n_usrs = self.MODEL.DATASET.DATA["N_USR_PCTG"]
            self.KHOT = MultiLabelBinarizer(classes=list(range(self.MODEL.DATASET.DATA["N_USR_PCTG"])))

            ret = []

            for id_r, rows in tqdm(self.MODEL.DATASET.DATA['TRAIN'].groupby("id_restaurant"), desc="Sequence data"):
                # Obtener el número de imágenes del restaurante
                rst_imgs = self.MODEL.DATASET.DATA["IMG"].loc[self.MODEL.DATASET.DATA["IMG"].reviewId.isin(rows.reviewId)].id_img.values
                # Asegurarse de que siempre tiene imágenes...
                assert len(rst_imgs)>0
                # Obtener usuarios
                rltd_usrs = rows.id_user.unique()
                # Asegurarse de que los ids son del X%
                # assert sum(rltd_usrs<n_usrs) == len(rltd_usrs)
                # Asegurarse que el número mínimo de usuarios de cada restaurante es el deseado.
                assert len(rltd_usrs) >= self.MODEL.CONFIG["data_filter"]["min_usrs_per_rest"]
                # Crear ejemplos
                r = [id_r] * len(rst_imgs)
                rn = [rows.rest_name.values[0]] * len(rst_imgs)
                x = rst_imgs
                # y = [list(rltd_usrs)] * len(rst_imgs)
                y = [list(rltd_usrs[rltd_usrs < n_usrs])] * len(rst_imgs) # En la salida, solo aquellos usuarios que son del 25%

                ret.extend(list(zip(r, rn, x, y)))

            ret = pd.DataFrame(ret, columns=["id_restaurant", "rest_name", "id_img", "output"]).sample(frac=1)

            ############################################################################################################

            self.ALL_DATA = ret

            if (len(ret) > self.MODEL.CONFIG["model"]["batch_size"]):
                self.BATCH = np.array_split(ret, len(ret) // self.MODEL.CONFIG["model"]["batch_size"])

            else:
                self.BATCH = np.array_split(ret, 1)

        def __len__(self):
            return len(self.BATCH)

        def __getitem__(self, idx):

            btch = self.BATCH[idx]

            x = self.MODEL.DATASET.DATA['IMG_VEC'][btch.id_img.values]
            y = self.KHOT.fit_transform(btch.output.values)

            return (x, y)

    ####################################################################################################################

    def __get_model__(self, layer_name ="img_emb"):

        self.MODEL.load_weights(self.MODEL_PATH + "weights")
        sub_model = Model(inputs=[self.MODEL.get_layer("in").input], outputs=[self.MODEL.get_layer(layer_name).output])

        return sub_model

