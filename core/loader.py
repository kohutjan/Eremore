import rawpy


class Loader:
    def __init__(self):
        pass

    @staticmethod
    def load(path):
        with rawpy.imread(path) as image:
            raw_image = image.raw_image.copy()
        return raw_image
