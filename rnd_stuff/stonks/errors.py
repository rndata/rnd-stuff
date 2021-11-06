class EnvError(Exception):
    pass


class Inconsistent(EnvError):
    pass


class DatasetError(EnvError):
    pass
