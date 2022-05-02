import warnings

from config.default_params import DEFAULT_PARAMS


class ParamReader:

    def __init__(self, params):
        self.param_dict = params

    def get_param(self, key):
        if key in self.param_dict:
            val = self.param_dict[key]
        else:
            val = DEFAULT_PARAMS[key]
            warnings.warn("Using default parameter for: {} with value: {}".format(key, val))

        return val
