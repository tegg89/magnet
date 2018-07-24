from models.graph_generation.transformer import *


def model_NN2(features, labels, mode):
    # Implement 2 seperate preprocessing of state and graph and combine them together

    # implementation of NN with 3 inputs: https://stackoverflow.com/questions/40318457/how-to-build-a-multiple-input-graph-with-tensor-flow

    # First state preprocessing
    input_layer_state1 = tf.reshape(features["state"], [-1, 38, 11, 1])
    conv1_state1 = tf.layers.conv2d(
        inputs=input_layer_state1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1_state1 = tf.layers.max_pooling2d(inputs=conv1_state1, pool_size=[2, 2], strides=2)
    conv2_state1 = tf.layers.conv2d(
        inputs=pool1_state1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2_state1 = tf.layers.max_pooling2d(inputs=conv2_state1, pool_size=[2, 2], strides=2)
    pool2_flat_state1 = tf.reshape(pool2_state1, [-1, 9 * 2 * 64])

    # Graph preprocessing
    input_layer = tf.reshape(features["graph"], [-1, 4, 120, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    conv2_flat = tf.reshape(conv2, [-1, 4 * 30 * 64])

    whole_model = tf.concat([pool2_flat_state1, conv2_flat], 1)

    dense = tf.layers.dense(inputs=whole_model, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions["classes"])
    else:
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions)

    return spec
