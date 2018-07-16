import pommerman
from pommerman import agents
from enum import Enum
import numpy as np
import tensorflow as tf
#import tensorflow as tf№from tensorflow.python import debug as tf_debug

# contain dictionary which describe current bombs on field
list_of_sest_boobms = [[],[],[],[]] 
#contain dictionary of dead agent, coordinates of bombs which kill them and flag 'was dead on privius step'
killed = {} 
# dictionary of coordinate_of_adgent
coordinate_of_adgent = {0: [0, 0],
                        1: [0, 0],
                        2: [0, 0],
                        3: [0, 0]
                        }

def init_list_of_vertex(board):
    vertex_name = {} # dictionary of coordinate (converted to x*11 + y, except agents: they alwas stay in 121, 122,
    # 123 and 124  + converted position) to type of vertex
    vertex_list = [] # list of existed vertex in current state
    for i in range(11):
        for j in range(11):
            if board[i,j] == 2:
                vertex_name[i * 11 + j] = 'wooden wall'
                vertex_list.append(i * 11 + j)
            if board[i,j] == 3:
                vertex_name[i * 11 + j] = 'bomb'
                vertex_list.append(i * 11 + j)
            if board[i,j] == 4:
                vertex_name[i * 11 + j] = 'flames'
                vertex_list.append(i * 11 + j)
            if board[i,j] == 5:
                vertex_name[i * 11 + j] = 'fog'
                vertex_list.append(i * 11 + j)
            if board[i,j] == 6:
                vertex_name[i * 11 + j] = 'extra bomb'
                vertex_list.append(i * 11 + j)
            if board[i,j] == 7:
                vertex_name[i * 11 + j] = 'increase range'
                vertex_list.append(i * 11 + j)
            if board[i,j] == 8:
                vertex_name[i * 11 + j] = 'can kick Power-Up'
                vertex_list.append(i * 11 + j)
            if board[i,j] == 10:
                vertex_name[121] = 0
                vertex_list.append(121)
            if board[i,j] == 11:
                vertex_name[122] = 1
                vertex_list.append(122)
            if board[i,j] == 12:
                vertex_name[123] = 2
                vertex_list.append(123)
            if board[i,j] == 13:
                vertex_name[124] = 3
                vertex_list.append(124)
    return vertex_name, vertex_list


def check_next_to_bomb(graph, agent_num, current_state, privius_vertex_name):
    for bomb in list_of_sest_boobms[agent_num]:
        bomb['blast strength'] = current_state[bomb['x'] + 23, bomb['y']]
        bomb['life'] = current_state[bomb['x'] + 1, bomb['y']]
        if (int(bomb['life']) == 1):
            for key, val in privius_vertex_name.items():
                item_x = key // 11
                item_y = key % 11
                #if it is an agent -- watch coordinet in coordinate_of_adgent
                if val in [0, 1, 2, 3] :
                    item_x, item_y = coordinate_of_adgent[val]

                if ((bomb['x'] - bomb['blast strength'] < item_x < bomb['x'] + bomb['blast strength']) and (
                        item_y == bomb['y'])) or \
                        ((bomb['y'] - bomb['blast strength'] < item_y < bomb['y'] + bomb['blast strength']) and (
                                item_x == bomb['x'])):

                    # kill someone
                    if val in [0, 1, 2, 3] and val != agent_num:
                        print(agent_num, " kill ", val)
                        killed[val]={'x':bomb['x'],
                                     'y':bomb['y'],
                                     'was dead on privius step': False
                        }

                    # kill yourself
                    if val == agent_num:
                        print(agent_num, " kill itself")
                        killed[val] = {'x': bomb['x'],
                                       'y': bomb['y'],
                                       'was dead on privius step': False
                        }

                        # destroy wooden wall
                    if val == 'wooden wall':
                        print(agent_num, " destroyed wooden wall")

            list_of_sest_boobms[agent_num].remove(bomb)
    return graph


def reward_shaping(graph, current_state, privius_state, agent_num):
    coordinate_of_adgent[agent_num] = (int(current_state[0, 0]), int(current_state[0, 1]))
    privius_state = np.asmatrix(privius_state).reshape(38, 11)
    current_state = np.asmatrix(current_state).reshape(38, 11)
    current_vertex_name, current_vertex_list = init_list_of_vertex(current_state[12:23])
    privius_vertex_name, privius_vertex_list = init_list_of_vertex(privius_state[12:23])
    privius_x = int(privius_state[0,0])
    privius_y = int(privius_state[0,1])

    # on privius state agent lay bomb
    if privius_state[37, 0] == 5:
        for key,val in privius_vertex_name.items():
            if val == 'bomb': # consider all setted bombs
                bomb_x = key // 11
                bomb_y = key % 11
                # check if that bomb was next to adgent in privius state
                if abs(bomb_x - privius_x) + abs(bomb_y - privius_y) == 1:
                    list_of_sest_boobms[agent_num].append({'x' : bomb_x,
                                                           'y' : bomb_y,
                                                           'life' : current_state[bomb_x + 1, bomb_y],
                                                            'blast strength' : current_state[bomb_x + 23, bomb_y]})
                    # add list of [x, y, bomb life, boobm blast strength]


    # increase can kick
    if privius_state[34, 0] < current_state[34, 0]:
        print(agent_num, " increase can kick")
        graph[agent_num, privius_x * 11 + privius_y] = 10 # set edge between can kick power up and adjent as 10

    # increase ammo
    if privius_state[35, 0] < current_state[35, 0]:
        print(agent_num, " increase ammo")
        graph[agent_num, privius_x * 11 + privius_y] = 10  # set edge between increase ammo power up and adjent as 10

    # increase blast power
    if privius_state[36, 0] < current_state[36, 0]:
        print(agent_num, " increase blast power")
        graph[agent_num, privius_x * 11 + privius_y] = 10  # set edge between blast power power up and adjent as 10

    graph = check_next_to_bomb(graph, agent_num, current_state, privius_vertex_name)


    # has died
    if agent_num in killed and not killed[agent_num]['was dead on privius step']:
        print(agent_num, " has dead")
        killed[agent_num]['was dead on privius step'] = 'True'
        return graph

    return graph

def state_to_matrixe_with_action(obs, action):
    # In this implementation I just concatenate everything in one big matrix

    # for e in obs['enemies']:
    #    print(e)
    #    print(Item(e))
    # TODO enemies
    my_position = np.asmatrix(obs['position'])
    bomb_life = np.asmatrix(obs['bomb_life'])
    board = np.asmatrix(obs['board'])
    bombs = np.asmatrix(obs['bomb_blast_strength'])
    # enemies = np.asmatrix([Item_en(e) for e in obs['enemies']])
    can_kick = np.asmatrix(int(1 if obs['can_kick'] else 0))
    ammo = np.asmatrix(int(obs['ammo']))
    blast_strength = np.asmatrix(int(obs['blast_strength']))

    m = np.max([my_position.shape[1], bomb_life.shape[1], board.shape[1], bombs.shape[1], can_kick.shape[1], ammo.shape[1],
         blast_strength.shape[1]])

    my_position1 = np.concatenate((my_position, np.zeros((my_position.shape[0], m - my_position.shape[1]))), axis=1)
    bomb_life1 = np.concatenate((bomb_life, np.zeros((bomb_life.shape[0], m - bomb_life.shape[1]))), axis=1)
    board1 = np.concatenate((board, np.zeros((board.shape[0], m - board.shape[1]))), axis=1)
    bombs1 = np.concatenate((bombs, np.zeros((bombs.shape[0], m - bombs.shape[1]))), axis=1)
    # enemies1 = np.concatenate((enemies, np.zeros((enemies.shape[0], m - enemies.shape[1]))), axis=1)
    can_kick1 = np.concatenate((can_kick, np.zeros((can_kick.shape[0], m - can_kick.shape[1]))), axis=1)
    ammo1 = np.concatenate((ammo, np.zeros((ammo.shape[0], m - ammo.shape[1]))), axis=1)
    blast_strength1 = np.concatenate((blast_strength, np.zeros((blast_strength.shape[0], m - blast_strength.shape[1]))), axis=1)
    action = np.asmatrix(action)
    action1 = np.concatenate((action, np.zeros((action.shape[0], m - action.shape[1]))), axis=1)
    result = np.concatenate((my_position1, bomb_life1, board1, bombs1, can_kick1, ammo1, blast_strength1, action1),axis=0)
    return np.asmatrix(result)


def model_NN1(features, labels, mode):
    #Implement 3 seperate preprocessing and combine them together
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

    dense = tf.layers.dense(inputs=whole_model, units=1024, activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 1024]
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=480) # shape of matrix -- 30 * 4

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

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
    pool2_flat_state1 = tf.reshape(pool2_state1, [1, 9 * 2 * 64])

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
    conv2_flat = tf.reshape(conv2, [1, 4 * 30 * 64])
    whole_model = tf.concat([pool2_flat_state1, conv2_flat], 1)

    dense = tf.layers.dense(inputs=whole_model, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions["classes"])

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions)


def main():
    # Print all possible environments in the Pommerman registry
    # print(pommerman.registry)
    sess = tf.Session()
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

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
    NN1 = tf.estimator.Estimator(model_fn=model_NN1, model_dir="/tmp/mnist_convnet_modelNN12")
    # Set up logging for predictions
    tensors_to_logNN1 = {"probabilities": "softmax_tensor"}
    logging_hookNN1 = tf.train.LoggingTensorHook(tensors=tensors_to_logNN1, every_n_iter=50)

    # Create the Estimator
    NN2 = tf.estimator.Estimator(model_fn=model_NN2, model_dir="/tmp/mnist_convnet_modelNN22")
    # Set up logging for predictions
    tensors_to_logNN2 = {"probabilities": "softmax_tensor"}
    logging_hookNN2 = tf.train.LoggingTensorHook(tensors=tensors_to_logNN2, every_n_iter=50)

    r_sum = np.zeros(1)
    for i in range(1):
        # Make the "Free-For-All" environment using the agent list
        env.reset()
        # Run the episodes just like OpenAI Gym
        for i_episode in range(1):
            state = env.reset()
            done = False
            current_state = None
            privius_state = None
            graph = np.random.rand(4, 120).astype("float32") + 0.0001
            pr_action = None
            pr_pr_action = None
            while not done:
                env.render()
                actions = env.act(state)
                state, reward, done, info = env.step(actions)
                r_sum[i] = +reward[0]
                # as basic implementation I consider only one agent

                privius_state = current_state
                current_state = state

                if not (pr_pr_action is None):
                    # Train the model
                    for agent_num in range(4):
                        current_state_matrix = np.resize(state_to_matrixe_with_action(current_state[agent_num], action=pr_action[agent_num]).astype("float32"), (1, 38 * 11))
                        privius_state_matrix = np.resize(state_to_matrixe_with_action(privius_state[agent_num],action=pr_pr_action[agent_num]).astype("float32"), (1, 38 * 11))

                        reward_shaping(graph, current_state_matrix, privius_state_matrix, agent_num)

                        train_input_NN2 = tf.estimator.inputs.numpy_input_fn(
                            x={"state": current_state_matrix,
                               "graph": np.resize(graph, (1, 4 * 120))},
                            y=np.asarray([actions[agent_num]]),
                            batch_size=1,
                            num_epochs=None,
                            shuffle=True)

                        train_input_NN1 = tf.estimator.inputs.numpy_input_fn(
                            x={"state1": privius_state_matrix,
                               "state2": current_state_matrix},
                            y=np.asmatrix(graph.flatten()),
                            batch_size=1,
                            num_epochs=None,
                            shuffle=True)

                        # NN1.train(
                        #     input_fn=train_input_NN1,
                        #     steps=200,
                        #     hooks=[logging_hookNN1])
                        #
                        # NN2.train(
                        #     input_fn=train_input_NN2,
                        #     steps=200,
                        #     hooks=[logging_hookNN1])
                        # predictions = NN2.predict(input_fn=train_input_NN2)
                        # next_action = np.array(list(p['classes'] for p in predictions))

                pr_pr_action = pr_action
                pr_action = actions

            if i_episode > 300:
                break
        print('Game {} finished'.format(i))

    np.savetxt('result_2simple_2random.csv', r_sum, fmt='%1.4e')
    env.close()


if __name__ == '__main__':
    main()
