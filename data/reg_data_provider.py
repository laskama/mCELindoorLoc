import numpy as np
from sklearn.preprocessing import StandardScaler

from data.base_data_provider import BaseDataProvider


class RegDataProvider(BaseDataProvider):

    def __init__(self, params, dc, floor_height=5):
        super(RegDataProvider, self).__init__(params, dc)

        self.pos_scaler: StandardScaler = None
        self.floor_height = floor_height

    def get_output_dim(self):
        return 3

    def set_labels(self, scale_labels=False):
        y = np.zeros((len(self.floor), 3))
        y[:, :2] = self.pos
        y[:, 2] = self.floor * self.floor_height

        if scale_labels:
            pos_scaler = StandardScaler()
            pos_scaler.fit(self.get_data(y, 'train'))
            y = pos_scaler.transform(y)
            self.pos_scaler = pos_scaler

        self.y = y

        return self

    def calc_performance(self, y_pred, compute_error_vec=False):
        metrics = {}

        y_true = self.get_y('test')
        floor_true = self.get_data(self.floor, partition='test')

        if self.pos_scaler is not None:
            # inverse scaling of predictions
            y_pred = self.pos_scaler.inverse_transform(y_pred)
            y_true = self.pos_scaler.inverse_transform(y_true)

        floor_pred = np.round(y_pred[:, 2] / self.floor_height, decimals=0)
        corr_idx = np.where(floor_pred == floor_true)[0]

        floor_acc = len(corr_idx) / len(y_true)

        pos_error_vec = np.linalg.norm(y_pred[:, :2] - y_true[:, :2], axis=1)
        pos_error_mean = np.mean(pos_error_vec)
        pos_error_median = np.median(pos_error_vec)

        metrics['floor_ACC'] = floor_acc

        metrics['MSE'] = pos_error_mean
        metrics['MSE (median)'] = pos_error_median

        if compute_error_vec:
            metrics['error_vec'] = pos_error_vec

        return metrics, y_pred[:, :2], floor_pred
