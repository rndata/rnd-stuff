class StonksError(Exception):
    pass


class DatasetError(StonksError):
    pass


class DataSpaceMismatch(DatasetError):
    def __init__(self, space, data):
        self.space = space
        self.data = data
        super().__init__(f"{space} doesn't match data {data}")
