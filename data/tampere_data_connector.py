import numpy as np
import pandas as pd

from data.data_connector import DatasetConnector
from utils.definitions import get_project_root

root = get_project_root()


class TampereDataConnector(DatasetConnector):

    def __init__(self, floors=None):
        super().__init__()
        if floors is None:
            self.floors = [1, 2, 3, 4, 5]
        else:
            self.floors = floors

    def load_dataset(self):

        x_train = pd.read_csv(root + "/datasets/tampere/Training_rss_21Aug17.csv", delimiter=',', header=None)
        x_test = pd.read_csv(root + "/datasets/tampere/Test_rss_21Aug17.csv", delimiter=',', header=None)

        y_train = pd.read_csv(root + "/datasets/tampere/Training_coordinates_21Aug17.csv", delimiter=',', header=None)
        y_test = pd.read_csv(root + "/datasets/tampere/Test_coordinates_21Aug17.csv", delimiter=',', header=None)

        # set RSS values
        self.rss = np.concatenate((x_train, x_test))

        # set floor and position
        y = np.concatenate((y_train, y_test))

        self.pos = y[:, :2]

        # replace heights with floor idx (starting with 1st floor)
        y_h = y[:, 2]
        heights = np.sort(np.unique(y_h))
        self.num_floors = len(heights)

        self.floorplan_width = [200.0] * self.num_floors
        self.floorplan_height = [80.0] * self.num_floors

        for idx, h in enumerate(heights):
            floor_idx = self.floors[idx] # idx + 1
            y_h[np.where(y_h == h)[0]] = floor_idx

        self.floor = y_h.astype(int)

        # set split indices based on predetermined split
        self.split_indices = [{"train": np.arange(0, len(x_train)),
                               "val": [],
                               "test": np.arange(len(x_train),
                                                 len(x_train) + len(x_test))}]

        return self

    def get_dataset_identifier(self):
        return 'tampere'


if __name__ == '__main__':
    dp = TampereDataConnector().load_dataset()
    print("test")