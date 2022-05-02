import tensorflow as tf


def get_model_from_yaml_definition(conf, input_dim, output_dim):

    input = tf.keras.layers.Input(shape=input_dim, name='input')
    bb = input

    # generate input based on backbone type
    bb_conf = conf['backbone']
    if bb_conf is not None:
        bb_type = bb_conf['type']

        if bb_type == "MLP":
            for l in bb_conf['layers']:
                bb = tf.keras.layers.Dense(l, activation=bb_conf['activation'])(bb)

                if 'dropout' in bb_conf:
                    bb = tf.keras.layers.Dropout(bb_conf['dropout'])(bb)

    # generate HEAD based on model type
    head = bb
    h_conf = conf['head']

    if conf['type'] == 'mCEL':
        # two output branches
        # (one for grid cell classification, one for regression)

        # classification head
        class_conf = h_conf['classification']

        c_head = head
        for l in class_conf['layers']:
            c_head = tf.keras.layers.Dense(l)(c_head)
            c_head = tf.keras.layers.Activation(class_conf['activation'])(c_head)

            if 'dropout' in class_conf:
                c_head = tf.keras.layers.Dropout(class_conf['dropout'])(c_head)

        c_output = tf.keras.layers.Dense(output_dim[0])(c_head)
        c_output = tf.keras.layers.Activation('softmax', name="output_class")(
            c_output)

        # regression head
        reg_conf = h_conf['regression']
        r_head = head
        for l in reg_conf['layers']:
            r_head = tf.keras.layers.Dense(l)(r_head)
            r_head = tf.keras.layers.Activation(reg_conf['activation'])(r_head)
            if 'dropout' in reg_conf:
                r_head = tf.keras.layers.Dropout(reg_conf['dropout'])(r_head)

        r_output = tf.keras.layers.Dense(output_dim[1])(r_head)
        r_output = tf.keras.layers.Activation('tanh', name="output_reg")(r_output)

        model = tf.keras.models.Model(input, [c_output, r_output])

    elif conf['type'] == '3D' or conf['type'] == '2D':
        # 3D regression model

        # regression head
        for l in h_conf['layers']:
            head = tf.keras.layers.Dense(l)(head)
            head = tf.keras.layers.Activation(activation=h_conf['activation'])(head)

            if 'dropout' in h_conf:
                head = tf.keras.layers.Dropout(h_conf['dropout'])(head)

        output = tf.keras.layers.Dense(output_dim)(head)
        output = tf.keras.layers.Activation('linear')(output)

        model = tf.keras.models.Model(input, output)

    return model
