import yaml


class DictToDot(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        super().__init__()
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DictToDot(value)
            self[key] = value


def load_config(filename):
    docs = yaml.load_all(open(filename, 'r'))
    ret_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            ret_dict[k] = v
    return ret_dict


class Config(DictToDot):

    def __init__(self, filename='config/config.yaml'):
        super(DictToDot, self).__init__()
        config_dict = load_config(filename)
        config_dot = DictToDot(config_dict)
        for k, v in config_dot.items():
            setattr(self, k, v)

    __getattr__ = DictToDot.__getitem__
    __setattr__ = DictToDot.__setitem__
    __delattr__ = DictToDot.__delitem__


param = Config()
