import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from data.data_connector import DatasetConnector
from utils.definitions import get_project_root

root = get_project_root() + "/datasets/giaIndoorLoc"


class GiaVSLAMdataConnector(DatasetConnector):

    def __init__(self, floors=None, devices=None, test_trajectories=None, test_devices=None):
        super().__init__()

        self.devices = devices
        self.floors = floors
        self.test_devices = test_devices
        self.test_trajectories = test_trajectories
        self.mac_addr = None

        if self.devices is None:
            self.devices = []

        if self.floors is None:
            self.floors = []

    def _get_floor_dimensions(self):
        floor_dim_df = pd.read_csv(root + "/floor_dimensions.csv")
        self.floorplan_width, self.floorplan_height = [], []

        for floor_id in self.floors:
            row = floor_dim_df[floor_dim_df['id'] == floor_id]
            self.floorplan_width += row['width'].tolist()
            self.floorplan_height += row['height'].tolist()

        self.num_floors = len(self.floorplan_width)

    def _get_mac_addr_vector(self):
        dfs = []
        for floor in self.floors:
            for dev in self.devices:
                path = root + "/floor_{}/{}".format(floor, dev)
                if not os.path.exists(path):
                    continue

                dfs += [get_rss_df_of_phone(path)]

        df = pd.concat(dfs)

        self.mac_addr = np.unique(df["mac"].to_numpy())

    def _load_scan_based_dataset(self, normalize='minmax', test_trajectory=None, test_devices=None):

        self._get_floor_dimensions()
        self._get_mac_addr_vector()

        time = []
        device = []
        rss = []
        pos = []
        floor = []

        train_idx = []
        test_idx = []
        curr_idx = 0

        for f in self.floors:

            for dev in self.devices:
                path = root + "/floor_{}/{}".format(f, dev)

                if not os.path.exists(path):
                    continue

                # scan based data used for testing
                data_base, labels_base, trajectories, scan_ids, mac_addr, traj_ids, time_stamps = \
                    get_scan_based_rss_dataset_of_phone(dev=path, mac_addr=self.mac_addr)

                if test_devices is not None:
                    if dev in test_devices:
                        # add testing data
                        curr_idx = _add_new_data(rss, pos, floor, device, test_idx, data_base, labels_base, f, dev,
                                                 curr_idx, time, time_stamps)
                    else:
                        # add training data
                        curr_idx = _add_new_data(rss, pos, floor, device, train_idx, data_base, labels_base, f, dev,
                                                 curr_idx, time, time_stamps)

                    continue

                elif test_trajectory is not None:
                    test_sub_idx = []
                    train_sub_idx = []

                    for excl_traj in test_trajectory:
                        test = np.where(traj_ids == excl_traj)[0].tolist()
                        if len(test) > 0:
                            print(excl_traj)
                        train = np.where(traj_ids != excl_traj)[0].tolist()
                        test_sub_idx += test
                        train_sub_idx += train

                    train_sub_idx = np.unique(np.array(train_sub_idx)).tolist()
                    test_sub_idx = np.unique(np.array(test_sub_idx)).tolist()

                    data, labels = data_base[train_sub_idx], labels_base[train_sub_idx]
                    x_test, y_test = data_base[test_sub_idx], labels_base[test_sub_idx]
                    time_train, time_test = time_stamps[train_sub_idx], time_stamps[test_sub_idx]

                else:
                    # randomly select some scan based data
                    data, x_test, labels, y_test, _, traj_test, scan_id_train, scan_id_test, time_train, time_test = \
                        train_test_split(data_base, labels_base, trajectories, scan_ids, time_stamps, random_state=1,
                                         test_size=0.3)

                # add training data
                curr_idx = _add_new_data(rss, pos, floor, device, train_idx, data, labels, f, dev, curr_idx, time,
                                         time_train)

                # add testing data
                curr_idx = _add_new_data(rss, pos, floor, device, test_idx, x_test, y_test, f, dev, curr_idx, time,
                                         time_test)

        self.rss = np.concatenate(rss, axis=0)
        self.pos = np.concatenate(pos, axis=0)
        self.floor = np.concatenate(floor, axis=0)
        self.device = np.concatenate(device, axis=0)
        self.time = np.concatenate(time, axis=0)

        if normalize == 'minmax':
            min_val = np.min(self.rss)
            max_val = np.max(self.rss)

            self.rss = (self.rss - min_val) / (max_val - min_val)

        self.split_indices = [{
            "train": np.concatenate(train_idx, axis=0),
            "val": [],
            "test": np.concatenate(test_idx, axis=0)}]

        return self

    def load_dataset(self):
        self._load_scan_based_dataset(test_devices=self.test_devices, test_trajectory=self.test_trajectories)

        return self

    def get_dataset_identifier(self):
        return 'gia_vslam'


def _add_new_data(rss, pos, floor, device, split_idx, data, labels, f, dev, curr_idx, time=None, time_stamps=None):
    # add training data
    rss += [data]
    pos += [labels]
    floor += [np.array([f] * len(labels))]
    device += [np.array([dev] * len(labels))]

    if time_stamps is not None:
        time += [time_stamps]

    new_idx = np.arange(curr_idx, curr_idx + len(data))
    split_idx += [new_idx]
    curr_idx += len(new_idx)

    return curr_idx


#
#   Get annotated WiFi fingerprinting dataset (average position tag per scan)
#


def get_scan_based_rss_dataset_of_phone(dev="OnePlus_2", mac_addr=None):
    """
    Obtain labeled WiFi dataset by using the mean annotated positions of a single WiFi scan
    to globally annotate the entire scan (fingerprint)
    Args:
        path: The path to the collected trajectories
        mac_addr: optional numpy array that holds the mac_addr which should be used. If None => use sorted appearance of
                  mac addresses to generate RSS vector

    Returns:
    Tuple of (RSS matrix, Position matrix, trajectories, scan_ids (for each row of matrix), mac_addr (columns of matrix)
    """
    df = get_rss_df_of_phone(dev)

    # exclude scans without position
    null_pos_scans = df[["x_coord", "y_coord"]].isnull().any(axis=1)
    df = df[~null_pos_scans]

    if mac_addr is None:
        mac_addr = df["mac"].unique()

    scan_ids = df["id"].unique()

    data = np.full((len(scan_ids), len(mac_addr)), -110.0)
    labels = np.zeros((len(scan_ids), 2))
    trajectories = []
    timestamps = []
    trajectory_ids = []

    for idx, id in enumerate(scan_ids):
        sub = df[df["id"] == id]
        positions = sub[["x_coord", "y_coord"]].to_numpy()
        timestamps += sub[["time"]].mean().to_list()
        trajectories += [positions]
        pos_avg = np.mean(positions, axis=0)
        labels[idx, :] = pos_avg
        trajectory_ids += [df[df["id"] == id].iloc[0, -1]]
        for _, s in sub.iterrows():
            mac_idx = np.where(mac_addr == s["mac"])[0]
            data[idx, mac_idx] = s["rss"]

    return data, labels, trajectories, scan_ids, mac_addr, np.array(trajectory_ids), np.array(timestamps)


def get_rss_df_of_phone(path="OnePlus_2"):
    """
    Obtain annotated WiFi scans as dataframe
    Args:
        path: The path to the collected trajectories

    Returns:
    Dataframe that holds annotated WiFi scans
    """
    id_offset = 0
    dfs = []
    for d in os.listdir(path):
        fp = path + "/" + d

        if not os.path.exists(fp + "/wifi_annotated.csv"):
            continue

        df = pd.read_csv(fp + "/wifi_annotated.csv")
        df["id"] += id_offset
        id_offset = df["id"].max(axis=0)
        dfs += [df]
    df = pd.concat(dfs, axis=0)

    df = df.sort_values(by=["time", "id"])

    return df


if __name__ == '__main__':
    dp = GiaVSLAMdataConnector(floors=[4], devices=["LG_ref", "OnePlus_ref"]).load_dataset()
