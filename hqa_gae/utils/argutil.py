import yaml

class PowerDict(dict):
    def __init__(self, d, recursive=True):
        super().__init__(d)
        self.recursive = recursive
        if self.recursive:
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = PowerDict(v, recursive)
        self.__dict__.update(d)

    def update(self, d):
        super().update(d)
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = PowerDict(v, True)
        self.__dict__.update(d)
    def pop(self, key):
        super().pop(key)
        self.__dict__.pop(key)

    def to_yaml(self, path):
        with open(path, 'w', encoding="utf-8") as file:
            yaml.dump(self.to_dict(), file)

    def to_dict(self):
        dict_ = super().copy()
        if self.recursive:
            for k in self.__dict__:
                if isinstance(self.__dict__[k], PowerDict):
                    dict_[k] = self.__dict__[k].to_dict()
        return dict_


def parse_yaml(yaml_file, object_view=True):
    with open(yaml_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    if object_view:
        data = PowerDict(data)
    return data


def get_default_config():
    return PowerDict({
        "t0": 1.,
        "temp_decay": 0.,
        "lambda_": 2,
        "alpha": 1.,
        "beta": 0.1,
        "l2_norm": True,
        "num_neighbors": None,
        "batch_size": 1,
        "num_workers": 0,
        "vq_type": 1,
        "patience": 1000,
        "optim": {
            "optimizer": {
                "type": "adam",
                "lr": 0.01,
                "weight_decay": 0.
            },
            "scheduler": {
                "type": "cos",
                "opt_restart": 1000,
                "warmup": None
            } 
        }
    })