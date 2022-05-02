from data.reg_data_provider import RegDataProvider
from model.base_model import BaseModel
import tensorflow as tf

from model.model_definition import get_model_from_yaml_definition


class RegModel(BaseModel):

    def __init__(self, model_params, data_provider: RegDataProvider, base_dir, model=None, model_name='model'):
        super(RegModel, self).__init__(model_params, data_provider, base_dir, model, model_name)
        self.dp = data_provider

    def setup_model(self, model_params):

        input_dim = self.dp.get_input_dim()
        output_dim = self.dp.get_output_dim()

        # obtain the tensorflow model for the specified parameters
        model = get_model_from_yaml_definition(model_params, input_dim, output_dim)

        model.compile(
            loss=tf.keras.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.pr.get_param('lr'))
        )

        self.model = model

        return self

    def evaluate_model(self, load_weights=True, compute_error_vec=False):
        if load_weights:
            self.model.load_weights(self.base_dir + self.model_name + ".hdf5")

        y_pred = self.model.predict(self.dp.get_x('test'))

        metrics, polys, floor_pred = self.dp.calc_performance(y_pred, compute_error_vec=compute_error_vec)

        return metrics, polys, floor_pred
