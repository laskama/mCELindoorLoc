import tensorflow as tf

from model.base_model import BaseModel
from data.mcel_data_provider import MCELdataProvider
from model.model_definition import get_model_from_yaml_definition


class MCELmodel(BaseModel):

    def __init__(self, model_params, data_provider: MCELdataProvider, base_dir, model=None, model_name='model'):
        super(MCELmodel, self).__init__(model_params, data_provider, base_dir, model, model_name)
        self.dp = data_provider

    def setup_model(self, model_params):

        input_dim = self.dp.get_input_dim()
        output_dim = self.dp.get_output_dim()
        num_grid_cells = output_dim[0]

        # obtain the tensorflow model for the specified parameters
        model = get_model_from_yaml_definition(model_params, input_dim, output_dim)

        model.compile(
            loss={'output_class': tf.keras.losses.categorical_crossentropy,
                  'output_reg': get_mcel_loss(int(num_grid_cells))},
            loss_weights={'output_class': 1.0, 'output_reg': 1.0},
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.pr.get_param('lr'))
        )

        self.model = model

        return self

    def evaluate_model(self, load_weights=True, compute_error_vec=False):
        if load_weights:
            self.model.load_weights(self.base_dir + self.model_name + ".hdf5")

        y_pred_grid, y_pred_box = self.model.predict(self.dp.get_x('test'))

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
