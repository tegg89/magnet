import pommerman
from pommerman import agents
from enum import Enum
import numpy as np
import tensorflow as tf

class Item(Enum):
    """The Items in the game.

    When picked up:
      - ExtraBomb increments the agent's ammo by 1.
      - IncrRange increments the agent's blast strength by 1.
      - Kick grants the agent the ability to kick items.

    AgentDummy is used by team games to denote the third enemy and by ffa to
    denote the teammate.
    """
    Passage = 0
    Rigid = 1
    Wood = 2
    Bomb = 3
    Flames = 4
    Fog = 5
    ExtraBomb = 6
    IncrRange = 7
    Kick = 8
    AgentDummy = 9
    Agent0 = 10
    Agent1 = 11
    Agent2 = 12
    Agent3 = 13

def state_to_matrix(obs):
    #In this implementation I just concatenate everything in one big matrix
    def convert_bombs(bomb_map):
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            })
        print('convert_bomb_ret', ret)
        return ret

    #for e in obs['enemies']:
    #    print(e)
    #    print(Item(e))
    #TODO enemies
    my_position = np.asmatrix(obs['position'])
    bomb_life = np.array(obs['bomb_life'])
    board = np.array(obs['board'])
    bombs = np.asmatrix(convert_bombs(np.array(obs['bomb_blast_strength'])))
    #enemies = np.asmatrix([Item_en(e) for e in obs['enemies']])
    can_kick = np.asmatrix(int(1 if obs['can_kick'] else 0))
    ammo = np.asmatrix(int(obs['ammo']))
    blast_strength = np.asmatrix(int(obs['blast_strength']))

    m = np.max([my_position.shape[1], bomb_life.shape[1], board.shape[1], bombs.shape[1],  can_kick.shape[1], ammo.shape[1], blast_strength.shape[1]])

    my_position1 = np.concatenate((my_position, np.zeros(( my_position.shape[0], m - my_position.shape[1]))), axis=1)
    bomb_life1 = np.concatenate((bomb_life, np.zeros((bomb_life.shape[0], m - bomb_life.shape[1]))), axis=1)
    board1 = np.concatenate((board, np.zeros((board.shape[0], m - board.shape[1]))), axis=1)
    bombs1 = np.concatenate((bombs, np.zeros((bombs.shape[0], m - bombs.shape[1]))), axis=1)
    #enemies1 = np.concatenate((enemies, np.zeros((enemies.shape[0], m - enemies.shape[1]))), axis=1)
    can_kick1 = np.concatenate((can_kick, np.zeros((can_kick.shape[0], m - can_kick.shape[1]))), axis=1)
    ammo1 = np.concatenate((ammo, np.zeros((ammo.shape[0], m - ammo.shape[1]))), axis=1)
    blast_strength1 = np.concatenate((blast_strength, np.zeros((blast_strength.shape[0], m - blast_strength.shape[1]))), axis=1)

    result = np.concatenate((my_position1, bomb_life1, board1, board1, bombs1, can_kick1, ammo1, blast_strength1), axis=0)
    return result



def state_to_matrixes(obs):
    #In this implementation I put all vector in matrix and concatenate this matrixes
    def get_map(board, item):
        map = np.zeros(shape)
        map[board == item] = 1
        return map

    def convert_bombs(bomb_map):
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            })
        return ret

    board = np.array(obs['board'])

    rigid_map = get_map(board, 1)  # Rigid = 1
    wood_map = get_map(board, 2)  # Wood = 2
    bomb_map = get_map(board, 3)  # Bomb = 3
    flames_map = get_map(board, 4)  # Flames = 4
    fog_map = get_map(board, 5)  # Fog = 5
    extra_bomb_map = get_map(board, 6)  # ExtraBomb = 6
    incr_range_map = get_map(board, 7)  # IncrRange = 7
    kick_map = get_map(board, 8)  # Kick = 8
    skull_map = get_map(board, 9)  # Skull = 9

    my_position = np.asmatrix(tuple(obs['position']))
    bomb_life = np.array(obs['bomb_life'])
    bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
    enemies = np.asmatrix([constants.Item(e) for e in obs['enemies']])
    can_kick = np.asmatrix(int(obs['Can Kick']))
    ammo = np.asmatrix(int(obs['ammo']))
    blast_strength = np.asmatrix(int(obs['blast_strength']))

    m = np.max([my_position.shape[1], bomb_life.shape[1], board.shape[1], bombs.shape[1], enemies.shape[1], can_kick.shape[1], ammo.shape[1], blast_strength.shape[1]])

    my_position1 = np.concatenate((my_position, np.zeros(( my_position.shape[0], m - my_position.shape[1]))), axis=1)
    bomb_life1 = np.concatenate((bomb_life, np.zeros((bomb_life.shape[0], m - bomb_life.shape[1]))), axis=1)
    board1 = np.concatenate((board, np.zeros((board.shape[0], m - board.shape[1]))), axis=1)
    bombs1 = np.concatenate((bombs, np.zeros((bombs.shape[0], m - bombs.shape[1]))), axis=1)
    enemies1 = np.concatenate((enemies, np.zeros((enemies.shape[0], m - enemies.shape[1]))), axis=1)
    can_kick1 = np.concatenate((can_kick, np.zeros((can_kick.shape[0], m - can_kick.shape[1]))), axis=1)
    ammo1 = np.concatenate((ammo, np.zeros((ammo.shape[0], m - ammo.shape[1]))), axis=1)
    blast_strength1 = np.concatenate((blast_strength, np.zeros((blast_strength.shape[0], m - blast_strength.shape[1]))), axis=1)

    result = np.concatenate((my_position1, bomb_life1, board1, board1, bombs1, enemies1, can_kick1, ammo1, blast_strength1, rigid_map, wood_map,\
                             bomb_map, flames_map, fog_map, fog_map, extra_bomb_map, incr_range_map, kick_map, skull_map), axis=0)
    return result



def model_NN1(features, labels, mode):
    #Implement 3 seperate preprocessing and combine them together
    # implementation of NN with 3 inputs: https://stackoverflow.com/questions/40318457/how-to-build-a-multiple-input-graph-with-tensor-flow

    # First state preprocessing
    input_layer = tf.placeholder(tf.reshape(features["graph"], [-1, 4, 30, 1]), trainable = True)
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
    # Output Tensor Shape: [batch_size, 19, 5, 64]
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

    # Graph preprocessing
    input_layer = tf.reshape(features["graph"], [-1, 4, 30, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 4, 30, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Output Tensor Shape: [batch_size, 2, 15, 32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 2, 15, 64]
    conv2_flat = tf.reshape(conv2, [-1, 2 * 15 * 64])

    whole_model = tf.concat([pool2_flat_state2, pool2_flat_state1, conv2_flat], 1)

    dense = tf.layers.dense(inputs=whole_model, units=1024, activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 1024]
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=6)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=6)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def model_NN2(features, labels, mode):
    # Implement 2 seperate preprocessing of state and graph and combine them together

    # implementation of NN with 3 inputs: https://stackoverflow.com/questions/40318457/how-to-build-a-multiple-input-graph-with-tensor-flow

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
    # Output Tensor Shape: [batch_size, 19, 5, 64]
    pool2_state1 = tf.layers.max_pooling2d(inputs=conv2_state1, pool_size=[2, 2], strides=2)
    pool2_flat_state1 = tf.reshape(pool2_state1, [-1, 9 * 2 * 64])

    # Graph preprocessing
    input_layer = tf.reshape(features["graph"], [-1, 4, 30, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 4, 30, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Output Tensor Shape: [batch_size, 2, 15, 32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 2, 15, 64]
    conv2_flat = tf.reshape(conv2, [-1, 2 * 15 * 64])

    whole_model = tf.concat([pool2_flat_state1, conv2_flat], 1)

    dense = tf.layers.dense(inputs=whole_model, units=1024, activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 1024]
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=6)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=6)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.registry)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    env = pommerman.make('PommeFFACompetition-v0', agent_list)


    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_NN1, 
        model_dir="/tmp/mnist_convnet_model1")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


    r_sum = np.zeros(1)
    for i in range(1):
        # Make the "Free-For-All" environment using the agent list
        obs = env.reset()
        # Run the episodes just like OpenAI Gym
        for i_episode in range(1):
            state = env.reset()
            done = False
            current_state = obs[0]
            previous_state = obs[0]
            # graph = tf.placeholder(np.random.rand((4,30)))
            graph = np.random.rand(4,30).astype("float64")
            while not done:
                env.render()
                actions = env.act(state)
                obs, reward, done, _ = env.step(actions)
                r_sum[i] = reward[0]

                # as basic implementation I consider only one agent
                previous_state = current_state
                current_state = obs[0]

                # Train the model
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"state1": state_to_matrix(previous_state),
                       "state2": state_to_matrix(current_state),
                       "graph": graph},
                    y=actions[0],
                    batch_size=1,
                    num_epochs=None,
                    shuffle=True)

                classifier.train(
                    input_fn=train_input_fn,
                    steps=200,
                    hooks=[logging_hook])
            if i_episode > 300:
                break
        print('Game {} finished'.format(i))

    np.savetxt('result_2simple_2random.csv', r_sum, fmt='%1.4e')
    env.close()


if __name__ == '__main__':
    main()
