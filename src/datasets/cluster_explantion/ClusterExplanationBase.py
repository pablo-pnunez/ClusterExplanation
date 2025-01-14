from src.datasets.DatasetClass import *
import pandas as pd

class ClusterExplanationBase(DatasetClass):

    def __init__(self, config):
        DatasetClass.__init__(self, config=config)

    def __basic_filtering__(self):

        def dropMultipleVisits(data):
            # Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
            multiple = data.groupby(["userId", "restaurantId"])["reviewId"].max().reset_index(name="last_reviewId")
            return data.loc[data.reviewId.isin(multiple.last_reviewId.values)].reset_index(drop=True)

        ################################################################################################################

        IMG = pd.read_pickle(self.CONFIG["data_path"] + "img-hd-densenet.pkl")
        RVW = pd.read_pickle(self.CONFIG["data_path"] + "reviews.pkl")
        RST = pd.read_pickle(self.CONFIG["data_path"] + "restaurants.pkl")
        RST.rename(columns={"name": "rest_name"}, inplace=True)

        if "index" in RVW.columns:RVW = RVW.drop(columns="index")

        IMG['review'] = IMG.review.astype(int)
        RST["id"] = RST.id.astype(int)

        RVW["reviewId"] = RVW.reviewId.astype(int)
        RVW["restaurantId"] = RVW.restaurantId.astype(int)

        RVW = RVW.merge(RST[["id","rest_name"]], left_on="restaurantId", right_on="id", how="left")

        RVW["num_images"] = RVW.images.apply(lambda x: len(x))
        RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
        RVW = RVW.loc[(RVW.userId != "")]

        # ---------------------------------------------------------------------------------------------------------------
        # Quedarse con las reviews positivas (>30)
        # --------------------------------------------------------------------------------------------------------------

        if self.CONFIG["only_positives"]:
            RVW = RVW.loc[RVW["like"] == 1]
            IMG = IMG.loc[IMG.review.isin(RVW.reviewId.unique())]

        # ---------------------------------------------------------------------------------------------------------------
        # Añadir URL a imágenes
        # --------------------------------------------------------------------------------------------------------------

        IMG = IMG.merge(RVW[["reviewId", "restaurantId", "images"]], left_on="review", right_on="reviewId", how="left")
        IMG["url"] = IMG.apply(lambda x: x.images[x.image]['image_url_lowres'], axis=1)
        IMG = IMG[["reviewId","restaurantId","image","url","vector","comida"]]

        # ---------------------------------------------------------------------------------------------------------------
        # ELIMINAR REVIEWS QUE NO TENGAN FOTO
        # --------------------------------------------------------------------------------------------------------------

        # RVW = RVW.loc[RVW.num_images>0]

        # ---------------------------------------------------------------------------------------------------------------
        # Eliminar fotos que no sean de comida
        # --------------------------------------------------------------------------------------------------------------
             
        IMG_NO = IMG.loc[IMG.comida==0]
        IMG = IMG.loc[IMG.comida==1].reset_index(drop=True)

        # Eliminar las reviews que tienen todas sus fotos de "NO COMIDA"
        IMG_NO_NUM = IMG_NO.groupby("reviewId").image.count().reset_index(name="drop")
        IMG_NO_NUM = IMG_NO_NUM.merge(RVW.loc[RVW.reviewId.isin(IMG_NO_NUM.reviewId.values)][["reviewId", "num_images"]], on="reviewId")
        IMG_NO_NUM["delete"] = IMG_NO_NUM["drop"] == IMG_NO_NUM["num_images"]

        RVW = RVW.loc[~RVW.reviewId.isin(IMG_NO_NUM.loc[IMG_NO_NUM.delete == True].reviewId.values)]

        # En las otras se actualiza el número de fotos
        for _, r in IMG_NO_NUM.loc[IMG_NO_NUM.delete==False].iterrows():
            RVW.loc[RVW.reviewId == r.reviewId, "num_images"] = RVW.loc[RVW.reviewId == r.reviewId, "num_images"]-r["drop"]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Quedarse con ultima review de los usuarios en caso de tener valoraciones diferentes (mismo rest)
        # --------------------------------------------------------------------------------------------------------------

        RVW = dropMultipleVisits(RVW)
        IMG = IMG.loc[IMG.reviewId.isin(RVW.reviewId)]

        return RVW, IMG
