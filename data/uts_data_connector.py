from data.data_connector import DatasetConnector
import pandas as pd
import numpy as np

from utils.definitions import get_project_root


class UTSdataConnector(DatasetConnector):

    def __init__(self):
        super(UTSdataConnector, self).__init__()

        self._floor = None
        self._bld = None
        self.bld_floor_map = {}

    def load_walls(self):
        raise NotImplementedError

    def load_dataset(self):
        root = get_project_root()

        train = pd.read_csv(root + "/datasets/uts/UTS_training.csv")

        test = pd.read_csv(root + "/datasets/uts/UTS_test.csv")

        data = pd.concat((train, test), axis=0)

        self.pos = data[['Pos_x', 'Pos_y']].to_numpy()
        self.rss = data.iloc[:, :589].to_numpy()

        self._bld = np.ones(len(data))
        self._floor = data['Floor_ID'].to_numpy()

        self.get_floorplan_dimensions()

        # set split indices based on predetermined split
        self.split_indices = [{"train": np.arange(0, len(train)),
                               "val": [],
                               "test": np.arange(len(train),
                                                 len(train) + len(test))}]

        return self

    def get_floorplan_dimensions(self):
        bld_flr = np.concatenate((self._bld.reshape((-1, 1)), self._floor.reshape((-1, 1))), axis=1)
        self.floor = np.zeros(len(self.rss), dtype=int)
        w = []
        h = []
        for idx, cmb in enumerate(np.unique(bld_flr, axis=0)):
            mask = np.logical_and(self._bld == cmb[0], self._floor == cmb[1])
            sub_idx = np.where(mask)[0]
            min_vals = np.min(self.pos[sub_idx], axis=0)
            w += [np.max(self.pos[sub_idx][:, 0]) - min_vals[0]]
            h += [np.max(self.pos[sub_idx][:, 1]) - min_vals[1]]

            self.pos[sub_idx] -= min_vals
            self.bld_floor_map[str(cmb.tolist())] = (idx + 1)
            self.floor[sub_idx] = (idx + 1)

        self.floorplan_width = w
        self.floorplan_height = h

        self.floors = np.unique(self.floor)
        self.num_floors = len(self.floorplan_width)

    def get_dataset_identifier(self):
        return 'uts'


if __name__ == '__main__':
    dp = UTSdataConnector()
    dp.load_dataset()
    print("test")