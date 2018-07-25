from models.graph_construction.transformer import *


def model_NN1(features, labels, mode):
    # Implement 3 seperate preprocessing and combine them together
    # implementation of NN with 3 inputs: https://stackoverflow.com/questions/40318457/how-to-build-a-multiple-input-graph-with-tensor-flow

    if labels is None:
        labels = features["y"]

    # labels = tf.placeholder(tf.float32, shape=(1, 480))
    # First state preprocessing
    input_layer_state1 = tf.reshape(features["state1"], [-1, 38, 11, 1])
    conv1_state1 = tf.layers.conv2d(
        inputs=input_layer_state1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 38, 11, 32]
    pool1_state1 = tf.layers.max_pooling2d(inputs=conv1_state1, pool_size=[2, 2], strides=2)
    # Output Tensor Shape: [batch_size, 19, 6, 32]
    conv2_state1 = tf.layers.conv2d(
        inputs=pool1_state1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 9, 5, 64]
    pool2_state1 = tf.layers.max_pooling2d(inputs=conv2_state1, pool_size=[2, 2], strides=2)
    pool2_flat_state1 = tf.reshape(pool2_state1, [-1, 9 * 2 * 64])

    # Second state preprocessing
    # Outputs equal to state1
    input_layer_state2 = tf.reshape(features["state2"], [-1, 38, 11, 1])
    conv1_state2 = tf.layers.conv2d(
        inputs=input_layer_state2,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1_state2 = tf.layers.max_pooling2d(inputs=conv1_state2, pool_size=[2, 2], strides=2)
    conv2_state2 = tf.layers.conv2d(
        inputs=pool1_state2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2_state2 = tf.layers.max_pooling2d(inputs=conv2_state2, pool_size=[2, 2], strides=2)
    pool2_flat_state2 = tf.reshape(pool2_state2, [-1, 9 * 2 * 64])

    whole_model = tf.concat([pool2_flat_state2, pool2_flat_state1], 1)

    dense = tf.layers.dense(inputs=whole_model, units=1024, activation=tf.nn.relu)  # (1, 1024)

    ##################################################################
    ######################### SELF-ATTENTION #########################
    ##################################################################

    decoder_inputs = tf.concat((tf.ones_like(labels[:, :1]) * 2, labels[:, :-1]), -1)  # (1,480)

    # Encoder
    with tf.variable_scope("encoder"):

        ## Positional Encoding
        enc = positional_encoding(dense, num_units=512, zero_pad=False, scale=False, scope="enc_pe")

        ## Dropout
        enc = tf.layers.dropout(enc, rate=0.1, training=tf.convert_to_tensor(True))

        ## Blocks
        for i in range(6):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                enc = multihead_attention(queries=enc, keys=enc, num_units=512, num_heads=8,
                                          dropout_rate=0.1, is_training=True, causality=False)

                ### Feed Forward
                enc = feedforward(enc, num_units=[4 * 512, 512])

    # Decoder
    with tf.variable_scope("decoder"):

        ## Positional Encoding
        dec = positional_encoding(decoder_inputs, num_units=512, zero_pad=False, scale=False, scope="dec_pe")

        ## Dropout
        dec = tf.layers.dropout(dec, rate=0.1, training=tf.convert_to_tensor(True))

        ## Blocks
        for i in range(6):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ## Multihead Attention ( self-attention)
                dec = multihead_attention(queries=dec, keys=dec, num_units=512, num_heads=8, dropout_rate=0.1,
                                          is_training=True, causality=True, scope="self_attention")

                ## Multihead Attention ( vanilla attention)
                dec = multihead_attention(queries=dec, keys=dec, num_units=512, num_heads=8, dropout_rate=0.1,
                                          is_training=True, causality=False, scope="vanilla_attention")

                ## Feed Forward
                dec = feedforward(dec, num_units=[4 * 512, 512])

        # Final linear projection
        dec = tf.layers.dense(dec, 1024)
        dec = tf.reshape(dec, [1, -1])

    ##################################################################
    ##################################################################
    ##################################################################

    # Output Tensor Shape: [batch_size, 1024]
    dropout = tf.layers.dropout(inputs=dec, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=480)  # shape of matrix -- 120 * 4

    if mode == tf.estimator.ModeKeys.PREDICT:
        print('PREDICT')
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"graph": logits})
    else:
        print('TRAIN/EVAL')
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        eval_metrics = {"rmse": tf.metrics.root_mean_squared_error(labels, logits)}
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metrics)
    return spec
