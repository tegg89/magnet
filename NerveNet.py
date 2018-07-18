from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

tf.logging.set_verbosity(tf.logging.INFO)
NUM_HIDDEN= 20
vertexes_mask = np.zeros((1, 120))
vertexes_mask[0, 0], vertexes_mask[0, 3], vertexes_mask[0, 118] = 1, 1, 1


def criric_model(features, labels, mode):
    h = {}
    m = {}

    # First step: h_{u}^{0} = MLP_{item}(s_{u})
    # first 120 elemeent desxribe items s_{u} and 120:124 desribe agents state s_{u}
    input_layer_data = tf.reshape(features["data"], [-1, 124, 38 * 11, 1])

    # MLP_{0} for items:
    for i in range(vertexes_mask.shape[1]):
        # if vertex exist
        if vertexes_mask[0, i] == 1:
            input_layer_vertex_mlp0 = tf.reshape(input_layer_data[:, i, :], [-1, 38, 11, 1])
            # Output shape == batch_size, 38, 11, 1
            conv1_vertex_mlp0 = tf.layers.conv2d(
                inputs=input_layer_vertex_mlp0,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            # Output Tensor Shape: [batch_size, 38, 11, 32]
            pool1_vertex_mlp0 = tf.layers.max_pooling2d(inputs=conv1_vertex_mlp0, pool_size=[2, 2], strides=2)
            # Output Tensor Shape: [batch_size, 19, 6, 32]
            conv2_vertex_mlp0 = tf.layers.conv2d(
                inputs=pool1_vertex_mlp0,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            # Output Tensor Shape: [batch_size, 19, 5, 64]
            pool2_vertex_mlp0 = tf.layers.max_pooling2d(inputs=conv2_vertex_mlp0, pool_size=[2, 2], strides=2)
            pool2_flat_vertex_mlp0 = tf.reshape(pool2_vertex_mlp0, [-1, 9 * 2 * 64])
            dense_vertex_mlp0 = tf.layers.dense(inputs=pool2_flat_vertex_mlp0, units=1024, activation=tf.nn.relu)
            # Output Tensor Shape: [batch_size, 1024]
            dropout_vertex_mlp0 = tf.layers.dropout(inputs=dense_vertex_mlp0, rate=0.4,
                                                    training=mode == tf.estimator.ModeKeys.TRAIN)
            logits_vertex_mlp0 = tf.layers.dense(inputs=dropout_vertex_mlp0, units=7)

            # Define h_{u}^{0} for items
            h[i] = tf.nn.softmax(logits_vertex_mlp0, name="softmax_tensor")

    # MLP_{0} for agents:
    for i in range(120, 124):
        input_layer_agent_mlp0 = tf.reshape(input_layer_data[:, i, :], [-1, 38, 11, 1])
        # Output shape == batch_size, 38, 11, 1
        conv1_agent_mlp0 = tf.layers.conv2d(
            inputs=input_layer_agent_mlp0,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        # Output Tensor Shape: [batch_size, 38, 11, 32]
        pool1_agent_mlp0 = tf.layers.max_pooling2d(inputs=conv1_agent_mlp0, pool_size=[2, 2], strides=2)
        # Output Tensor Shape: [batch_size, 19, 6, 32]
        conv2_agent_mlp0 = tf.layers.conv2d(
            inputs=pool1_agent_mlp0,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        # Output Tensor Shape: [batch_size, 19, 5, 64]
        pool2_agent_mlp0 = tf.layers.max_pooling2d(inputs=conv2_agent_mlp0, pool_size=[2, 2], strides=2)
        pool2_flat_agent_mlp0 = tf.reshape(pool2_agent_mlp0, [-1, 9 * 2 * 64])
        dense_agent_mlp0 = tf.layers.dense(inputs=pool2_flat_agent_mlp0, units=1024, activation=tf.nn.relu)
        # Output Tensor Shape: [batch_size, 1024]
        dropout_agent_mlp0 = tf.layers.dropout(inputs=dense_agent_mlp0, rate=0.4,
                                               training=mode == tf.estimator.ModeKeys.TRAIN)
        logits_agent_mlp0 = tf.layers.dense(inputs=dropout_agent_mlp0, units=7)
        # Define h_{u}^{0} for agent
        h[i] = tf.nn.softmax(logits_agent_mlp0, name="softmax_tensor")

    # Second step: m_{u}^{t} = MLP_{messege}(h_{u}^{t-1})
    # MLP_{messege} for items:
    for i in range(vertexes_mask.shape[1]):
        # if vertex exist
        if vertexes_mask[0, i] == 1:
            input_layer_vertex = tf.reshape(tf.concat([h[120], h[121], h[122], h[123]], 0), [-1, 4, 7, 1])
            # Output shape == batch_size, 4, 7, 1
            conv1_vertex = tf.layers.conv2d(
                inputs=input_layer_vertex,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            # Output Tensor Shape: [batch_size, 4, 7, 32]
            pool1_vertex = tf.layers.max_pooling2d(inputs=conv1_vertex, pool_size=[2, 2], strides=2)
            # Output Tensor Shape: [batch_size, 2, 3, 32]
            conv2_vertex = tf.layers.conv2d(
                inputs=pool1_vertex,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            # Output Tensor Shape: [batch_size, 2, 3, 64]
            conv2_flat_vertex = tf.reshape(conv2_vertex, [-1, 2 * 3 * 64])
            dense_vertex = tf.layers.dense(inputs=conv2_flat_vertex, units=1024, activation=tf.nn.relu)
            # Output Tensor Shape: [batch_size, 1024]
            dropout_vertex = tf.layers.dropout(inputs=dense_vertex, rate=0.4,
                                               training=mode == tf.estimator.ModeKeys.TRAIN)
            logits_vertex = tf.layers.dense(inputs=dropout_vertex, units=7)
            # Define m_{u}^{t} for items
            m[i] = tf.nn.softmax(logits_vertex, name="softmax_tensor")

    # MLP_{messege} for agent:
    # 1. Create input message
    # 1.0 concatinate all items message -- h_{item}^{t-1}
    list_input_layer_agent = []
    for i in range(vertexes_mask.shape[1]):
        # if agent exist
        if vertexes_mask[0, i] == 1:
            list_input_layer_agent.append(h[i])

    for i in range(120, 124):
        # 1.1 concatinate all other agent message -- h_{item}^{t-1}
        list_input_layer_agent_i = list_input_layer_agent.copy()
        for j in range(120, 124):
            if i != j:
                list_input_layer_agent_i.append(h[j])

        input_layer_agent = tf.reshape(tf.concat(list_input_layer_agent_i, 0),
                                       [-1, int(np.sum(vertexes_mask)) + 3, 7, 1])
        # Output shape == batch_size, num of items + 3 , 7, 1
        # MPL{agent} for messege:
        conv1_agent = tf.layers.conv2d(
            inputs=input_layer_agent,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        # Output Tensor Shape: [batch_size, num of items + 3 , 7, 32]
        pool1_agent = tf.layers.max_pooling2d(inputs=conv1_agent, pool_size=[2, 2], strides=2)
        # Output Tensor Shape: [batch_size, (num of items + 3) // 3 , 3, 32]
        conv2_agent = tf.layers.conv2d(
            inputs=pool1_agent,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        # Output Tensor Shape: [batch_size, (num of items + 3) // 3 , 3, 64]
        conv2_flat_agent = tf.reshape(conv2_agent, [-1, int((np.sum(vertexes_mask) + 3) // 2) * 3 * 64])
        dense_agent = tf.layers.dense(inputs=conv2_flat_agent, units=1024, activation=tf.nn.relu)
        # Output Tensor Shape: [batch_size, 1024]
        dropout_agent = tf.layers.dropout(inputs=dense_agent, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits_agent = tf.layers.dense(inputs=dropout_agent, units=7)

        # Define m_{u}^{t} for agent
        m[i] = tf.nn.softmax(logits_agent, name="softmax_tensor")

    # Third step: h_{u}^{t} = LSTM(m_{u}^{t}, h_{u}^{t - 1})
    # Implementation:
    for i in range(vertexes_mask.shape[1] + 4):
        # if vertex exist
        if i >= vertexes_mask.shape[1] or vertexes_mask[0, i] == 1:
            print(tf.concat([m[i], h[i]], 1))
            input_layer_vertex_lstm = tf.reshape(tf.concat([m[i], h[i]], 1), [1, 14])
            num_classes = 6
            # Define weights
            weights = {'out': tf.Variable(tf.random_normal([NUM_HIDDEN, num_classes]))}
            biases = {'out': tf.Variable(tf.random_normal([num_classes]))}
            lstm_cell = rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0)
            # Get lstm cell output
            # outputs, states = rnn.static_rnn(lstm_cell, [input_layer_vertex_lstm], dtype=tf.float32)
            dense_vertex_lstm = tf.layers.dense(inputs=input_layer_vertex_lstm, units=64, activation=tf.nn.relu)
            logits_vertex_lstm = tf.layers.dense(inputs=dense_vertex_lstm, units=7)
            # Define h_{u}^{t} for items
            h[i] = tf.nn.softmax(logits_vertex_lstm, name="softmax_tensor")

    # End of NerveNet
    # Implementation of Estimator
    ans = tf.concat([h[120], h[121], h[122], h[123]], 1)
    predictions = {
        "classes": tf.argmax(input=ans, axis=1),
        "probabilities": ans
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=ans)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions["classes"])

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    train_data = np.random.rand(1, 124 * 38 * 11).astype("float32")  # Returns np.array
    train_labels = np.random.rand(1, 4 * 7).astype("float32")

    eval_labels = np.random.rand(1, 4 * 7).astype("float32")
    eval_data = np.random.rand(1, 120 * 38 * 11).astype("float32")

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=criric_model)
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"data": train_data},
        y=train_labels,
        batch_size=1,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])

    # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #  x={"state1": np.resize(np.random.rand(4, 30).astype("float32"), (1, 48*11)),
    #     "graph": np.resize(np.random.rand(4, 30).astype("float32"), (1, 4*30)) },
    #  y=np.asarray([1]),
    #  num_epochs=1,
    #  shuffle=False)
    # predictions = mnist_classifier.predict(input_fn=eval_input_fn)
    # print(predictions)
    # y_predicted = np.array(list(p['classes'] for p in predictions))
    # print(y_predicted, 'he3')


if __name__ == "__main__":
    tf.app.run()