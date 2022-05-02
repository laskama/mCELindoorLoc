import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

from data.data_connector import DatasetConnector
from utils.param_reader import ParamReader


class BaseDataProvider:

    def __init__(self, params, dc: DatasetConnector):
        # used for consistent train/test split
        self.seed = 1

        # reference to dataset connector
        self.dc = dc

        # unprocessed variables of dataset (will be accessed indirectly via dataset connector)
        self.rss = None
        self.pos = None
        self.floor = None
        self.time = None
        self.num_floors = None

        # list of floorplan dimensions per floor
        self.floorplan_width = None
        self.floorplan_height = None

        # data and labels that are used by model for training
        # the goal of a specific data provider implementation is to set these two variables
        # such that a call to model.fit(x, y) succeeds where x and y are obtained via the data provider
        self.x = None
        self.y = None

        # split indices are used for obtaining the required partition (train/val/test) of the dataset
        # [{'train': [], 'val': [], 'test': [])]
        self._split_indices = None

        # currently, only one train/test split (fold) is supported
        self.num_folds = 1
        self.split_idx = 0

        # reference to parameter reader that falls back to default value if parameter is not specified
        self.pr: ParamReader = ParamReader(params)

    def load_dataset(self):
        self.dc.load_dataset()
        return self

    # getters & setters
    @property
    def rss(self):
        return self.dc.rss

    @property
    def pos(self):
        return self.dc.pos

    @property
    def floor(self):
        return self.dc.floor

    @property
    def floors(self):
        return self.dc.floors

    @property
    def time(self):
        if hasattr(self.dc, 'time'):
            return self.dc.time
        else:
            return None

    @property
    def num_floors(self):
        return self.dc.num_floors

    @property
    def floorplan_width(self):
        return self.dc.floorplan_width

    @property
    def floorplan_height(self):
        return self.dc.floorplan_height

    # getters & setters
    @rss.setter
    def rss(self, val):
        self.dc.rss = val

    @pos.setter
    def pos(self, val):
        self.dc.pos = val

    @floor.setter
    def floor(self, val):
        self.dc.floor = val

    @floors.setter
    def floors(self, val):
        self.dc.floors = val

    @time.setter
    def time(self, val):
        self.dc.time = val

    @num_floors.setter
    def num_floors(self, val):
        self.dc.num_floors = val

    @floorplan_width.setter
    def floorplan_width(self, val):
        self.dc.floorplan_width = val

    @floorplan_height.setter
    def floorplan_height(self, val):
        self.dc.floorplan_height = val

    @property
    def split_indices(self):
        s = self._split_indices
        if s is None:
            s = self.dc.split_indices
        return s

    @split_indices.setter
    def split_indices(self, val):
        self._split_indices = val

    #
    # Access of data via split indices
    #

    def get_x(self, partition='train'):
        subset = self.split_indices[self.split_idx][partition]
        return self.x[subset]

    def get_y(self, partition='train'):
        subset = self.split_indices[self.split_idx][partition]
        return self.y[subset]

    def get_data(self, data, partition='train'):
        subset = self.split_indices[self.split_idx][partition]
        return data[subset]

    #
    # Preprocessing methods
    #

    def replace_missing_values(self, mis_val=100, replace_val=-110):
        self.rss[self.rss == mis_val] = replace_val
        return self

    def standardize_data(self, powed=False, standardize=False):
        if powed:
            return self.normalize_powed()
        elif standardize:
            scaler = StandardScaler()
            self.x = scaler.fit_transform(self.rss)
        else:
            min_ap_val = np.min(self.rss)
            max_ap_val = np.max(self.rss)
            self.x = (self.rss - min_ap_val) / (max_ap_val - min_ap_val)

        return self

    def normalize_powed(self, b=2.71828):
        arr = self.rss
        res = np.copy(arr).astype(np.float)

        zero_mask = np.logical_or(res > 50, res < -95)
        one_mask = res >= 0

        rest = ~np.logical_or(zero_mask, one_mask)

        res[zero_mask] = 0
        res[one_mask] = 1
        res[rest] = ((95 + res[rest]) / 95.0) ** b

        self.x = res

        return self

    def generate_validation_indices(self):
        r = self.pr.get_param('val')
        if r == 0.0:
            return self

        for f_idx in range(self.num_folds):

            train_val = self.split_indices[f_idx]['train']
            train, val = train_test_split(
                train_val, test_size=r, random_state=self.seed)

            self.split_indices[f_idx]['train'] = train
            self.split_indices[f_idx]['val'] = val

        return self

    def get_input_dim(self):
        return np.shape(self.x)[1]

    def get_output_dim(self):
        return np.shape(self.y)[1]

    def generate_split_indices(self, n_splits=5, overwrite=False):

        # split indices might be given by dataset (do not overwrite except if explicitly required)
        if self.dc.split_indices is not None and not overwrite:
            return self

        kf = KFold(n_splits=n_splits, shuffle=True,
                   random_state=self.seed)
        splits = []
        for s in list(kf.split(self.rss)):
            splits += [{"train": s[0], "val": [], "test": s[1]}]

        self.split_indices = splits

        self.num_folds = kf.get_n_splits(self.rss)

        return self

    def store(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as f:
                dp = pickle.load(f)
        except FileNotFoundError:
            return None

        return dp
