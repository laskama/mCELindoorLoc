import tensorflow as tf

from data.base_data_provider import BaseDataProvider
from utils.param_reader import ParamReader


class BaseModel:

    def __init__(self, model_params, data_provider, base_dir, model=None, model_name='model'):
        self.model: tf.keras.Model = model
        self.base_dir = base_dir
        self.pr = ParamReader(model_params)
        self.dp: BaseDataProvider = data_provider
        self.model_name = model_name

    def fit_model(self):
        model = self.model
        dp = self.dp
        pr = self.pr

        ckpt_file = self.base_dir + self.model_name + ".hdf5"

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            ckpt_file, monitor='val_loss',
            save_best_only=True, mode='auto', save_weights_only=True
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

        # model.summary()

        model.fit(dp.get_x('train'), dp.get_y('train'),
                  validation_data=(dp.get_x('val'), dp.get_y('val')),
                  batch_size=pr.get_param('batch_size'),
                  epochs=pr.get_param('epochs'),
                  callbacks=[checkpoint, stop_early],
                  verbose=pr.get_param('verbose'))
