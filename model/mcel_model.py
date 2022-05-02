import tensorflow as tf
import numpy as np

from model.base_model import BaseModel
from data.mcel_data_provider import MCELdataProvider


class MCELmodel(BaseModel):

    def __init__(self, model_params, data_provider: MCELdataProvider, base_dir, model=None, model_name='model'):
        super(MCELmodel, self).__init__(model_params, data_provider, base_dir, model, model_name)
        self.dp = data_provider

    def setup_model(self, model):

        pr = self.pr

        num_grid_cells = self.dp.get_output_dim()[0]

        model.compile(
            loss={'output_class': tf.keras.losses.categorical_crossentropy,
                  'output_reg': get_mcel_loss(int(num_grid_cells))},
            loss_weights={'output_class': 1.0, 'output_reg': 1.0},
            optimizer=tf.keras.optimizers.Adam(learning_rate=pr.get_param('lr'))
        )

        self.model = model

        return self

    def evaluate_model(self, load_weights=True, compute_error_vec=False):
        if load_weights:
            self.model.load_weights(self.base_dir + self.model_name + ".hdf5")

        y_pred_grid, y_pred_box = self.model.predict(self.dp.get_x('test'))

        # convert to same encoding scheme als DLB to apply similar conversion
        # FROM:
        #   y_pred_grid = (g_1, ..., g_G)
        #   y_pred_box = (cx_1, cy_1, ..., cx_G, cy_G)
        # TO: (setting w & h to 0, since only point estimation)
        # merged = (cx_1, cy_1, w_1, h_1, g_1, ..., cx_G, cy_G, w_G, h_G, g_G)
        y_pred_merged = np.zeros((len(y_pred_grid),
                                  y_pred_box.shape[1] * 2 + int(
                                      y_pred_grid.shape[1])))
        for b in range(y_pred_grid.shape[1]):
            y_pred_merged[:, b * 4 + b:(b + 1) * 4 + b] = \
                np.concatenate((y_pred_box[:, b * 2:(b + 1) * 2],
                                np.zeros((len(y_pred_box), 2))), axis=1)
            y_pred_merged[:, (b + 1) * 4 + b] = y_pred_grid[:, b]

        metrics, polys, floor_pred = self.dp.calc_performance(
            y_pred_grid=y_pred_grid,
            y_pred_box=y_pred_box,
            y_true_grid_enc=self.dp.get_data(self.dp.grid_labels, 'test'),
            y_true_pos=self.dp.get_data(self.dp.pos, 'test'),
            compute_error_vec=compute_error_vec,
            )

        return metrics, polys, floor_pred


def get_mcel_loss(num_grid_cells):
    def multi_loss(y_true, y_pred):

        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], num_grid_cells, 2))
        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], num_grid_cells, 3))
        g = y_true[:, :, 2]
        coords = y_true[:, :, :2]

        loss = tf.reduce_sum(tf.square(tf.subtract(y_pred, coords)), axis=2)

        loss = tf.multiply(loss, g)

        loss = tf.reduce_sum(loss, axis=1)

        return loss

    return multi_loss
