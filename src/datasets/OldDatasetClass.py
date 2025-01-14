
class OldDatasetClass:

    def __init__(self, config):
        self.CONFIG = config
        self.DATA = self.get_data()

    def get_data(self):
        # Retorna un diccionario con los datos
        raise NotImplementedError

