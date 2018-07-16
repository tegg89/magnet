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

    #for e in obs['enemies']:
    #    print(e)
    #    print(Item(e))
    #TODO enemies
    my_position = np.asmatrix(obs['position'])
    bomb_life = np.asmatrix(obs['bomb_life'])
    board = np.asmatrix(obs['board'])
    bombs = np.asmatrix(obs['bomb_blast_strength'])
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
    return np.asmatrix(result)

# def state_to_matrixes(obs):
#     #In this implementation I put all vector in matrix and concatenate this matrixes
#     def get_map(board, item):
#         map = np.zeros(shape)
#         map[board == item] = 1
#         return map

#     def convert_bombs(bomb_map):
#         ret = []
#         locations = np.where(bomb_map > 0)
#         for r, c in zip(locations[0], locations[1]):
#             ret.append({
#                 'position': (r, c),
#                 'blast_strength': int(bomb_map[(r, c)])
#             })
#         return ret

#     board = np.array(obs['board'])

#     rigid_map = get_map(board, 1)  # Rigid = 1
#     wood_map = get_map(board, 2)  # Wood = 2
#     bomb_map = get_map(board, 3)  # Bomb = 3
#     flames_map = get_map(board, 4)  # Flames = 4
#     fog_map = get_map(board, 5)  # Fog = 5
#     extra_bomb_map = get_map(board, 6)  # ExtraBomb = 6
#     incr_range_map = get_map(board, 7)  # IncrRange = 7
#     kick_map = get_map(board, 8)  # Kick = 8
#     skull_map = get_map(board, 9)  # Skull = 9

#     my_position = np.asmatrix(tuple(obs['position']))
#     bomb_life = np.array(obs['bomb_life'])
#     bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
#     enemies = np.asmatrix([constants.Item(e) for e in obs['enemies']])
#     can_kick = np.asmatrix(int(obs['Can Kick']))
#     ammo = np.asmatrix(int(obs['ammo']))
#     blast_strength = np.asmatrix(int(obs['blast_strength']))

#     m = np.max([my_position.shape[1], bomb_life.shape[1], board.shape[1], bombs.shape[1], enemies.shape[1], can_kick.shape[1], ammo.shape[1], blast_strength.shape[1]])

#     my_position1 = np.concatenate((my_position, np.zeros(( my_position.shape[0], m - my_position.shape[1]))), axis=1)
#     bomb_life1 = np.concatenate((bomb_life, np.zeros((bomb_life.shape[0], m - bomb_life.shape[1]))), axis=1)
#     board1 = np.concatenate((board, np.zeros((board.shape[0], m - board.shape[1]))), axis=1)
#     bombs1 = np.concatenate((bombs, np.zeros((bombs.shape[0], m - bombs.shape[1]))), axis=1)
#     enemies1 = np.concatenate((enemies, np.zeros((enemies.shape[0], m - enemies.shape[1]))), axis=1)
#     can_kick1 = np.concatenate((can_kick, np.zeros((can_kick.shape[0], m - can_kick.shape[1]))), axis=1)
#     ammo1 = np.concatenate((ammo, np.zeros((ammo.shape[0], m - ammo.shape[1]))), axis=1)
#     blast_strength1 = np.concatenate((blast_strength, np.zeros((blast_strength.shape[0], m - blast_strength.shape[1]))), axis=1)

#     result = np.concatenate((my_position1, bomb_life1, board1, board1, bombs1, enemies1, can_kick1, ammo1, blast_strength1, rigid_map, wood_map,\
#                              bomb_map, flames_map, fog_map, fog_map, extra_bomb_map, incr_range_map, kick_map, skull_map), axis=0)
#     return result

def state_to_matrix_with_action(obs, action):
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
#     print(my_position1.shape, bomb_life1.shape, board1.shape, bombs1.shape, can_kick1.shape, ammo1.shape,
#           blast_strength1.shape, action.shape, action1.shape)
    result = np.concatenate((my_position1, bomb_life1, board1, bombs1, can_kick1, ammo1, blast_strength1, action1),
                            axis=0)
    return np.asmatrix(result)

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
#         print('mean', mean, 'variance', variance)
        beta= tf.Variable(tf.zeros(params_shape))
#         print('beta', beta)
        gamma = tf.Variable(tf.ones(params_shape))
#         print('gamma', gamma)
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
#         print('normalized before', normalized)
        ##Because normalized has float64 type. Need to change to float32 to operate.
        normalized = tf.cast(normalized, tf.float32)
#         print('normalized after', normalized)
        outputs = gamma * normalized + beta
#         print('outputs', outputs)
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
            print('lookup_table', lookup_table)
            print('inputs', inputs)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        print('outputs', outputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs
    

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs



def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
#         print('outputs', outputs)
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)

def model_NN1(features, labels, mode):
    #Implement 3 seperate preprocessing and combine them together
    # implementation of NN with 3 inputs: https://stackoverflow.com/questions/40318457/how-to-build-a-multiple-input-graph-with-tensor-flow

    # First state preprocessing
    input_layer_state1 = tf.reshape(features["state1"], [-1, 49, 11, 1])
    conv1_state1 = tf.layers.conv2d(
        inputs=input_layer_state1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 49, 11, 32]
    pool1_state1 = tf.layers.max_pooling2d(inputs=conv1_state1, pool_size=[2, 2], strides=2)
    # Output Tensor Shape: [batch_size, 24, 6, 32]
    conv2_state1 = tf.layers.conv2d(
        inputs=pool1_state1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Output Tensor Shape: [batch_size, 24, 5, 64]
    pool2_state1 = tf.layers.max_pooling2d(inputs=conv2_state1, pool_size=[2, 2], strides=2)
    pool2_flat_state1 = tf.reshape(pool2_state1, [-1, 12 * 2 * 64])

    # Second state preprocessing
    # Outputs equal to state1
    input_layer_state2 = tf.reshape(features["state2"], [-1, 49, 11, 1])
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
    pool2_flat_state2 = tf.reshape(pool2_state2, [-1, 12 * 2 * 64])

    whole_model = tf.concat([pool2_flat_state2, pool2_flat_state1], 1) # (1,3072)
    
    dense = tf.layers.dense(inputs=whole_model, units=1024, activation=tf.nn.relu) #(1, 1024)

    #################################################################
    #################################################################
    #################################################################
    
    decoder_inputs = tf.concat((tf.ones_like(labels[:, :1])*2, labels[:, :-1]), -1) #(1,120)
# #     print('decoder_inputs', decoder_inputs)


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
                enc = feedforward(enc, num_units=[4*512, 512])
            
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
                                          is_training=True,causality=True, scope="self_attention")

                ## Multihead Attention ( vanilla attention)
                dec = multihead_attention(queries=dec, keys=dec, num_units=512, num_heads=8, dropout_rate=0.1,
                                          is_training=True,causality=False, scope="vanilla_attention")

                ## Feed Forward
                dec = feedforward(dec, num_units=[4*512, 512])

        # Final linear projection
        dec = tf.layers.dense(dec, 1024)
        dec = tf.reshape(dec, [1, -1])
        
    #################################################################
    #################################################################
    #################################################################
    
    # Output Tensor Shape: [batch_size, 1024]
    dropout = tf.layers.dropout(inputs=dec, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    print('dropout', dropout) #(1, 120, 1024)

    logits = tf.layers.dense(inputs=dropout, units=120) # shape of matrix -- 30 * 4

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    print('labels', labels, 'logits', logits) #labels: (1,120), logits: (1,120,120)
    
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
    input_layer_state1 = tf.reshape(features["state1"], [-1, 49, 11, 1])
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
    pool2_flat_state1 = tf.reshape(pool2_state1, [-1, 12 * 2 * 64])

    # Graph preprocessing
    input_layer = tf.reshape(features["graph"], [-1, 4, 30, 1])
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
    conv2_flat = tf.reshape(conv2, [-1, 2 * 15 * 64])

    whole_model = tf.concat([pool2_flat_state1, conv2_flat], 1)

    dense = tf.layers.dense(inputs=whole_model, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
#     print('H')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
#     print(loss.shape, onehot_labels.shape)

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
	print(pommerman.registry)
	# sess = tf.Session()
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
	estimator_nn1 = tf.estimator.Estimator(model_fn=model_NN1, model_dir="/tmp/sa_nn1")
	# Set up logging for predictions
	tensors_to_logNN1 = {"probabilities": "softmax_tensor"}
	logging_hook_nn1 = tf.train.LoggingTensorHook(tensors=tensors_to_logNN1, every_n_iter=50)

	# Create the Estimator
	estimator_nn2 = tf.estimator.Estimator(model_fn=model_NN2, model_dir="/tmp/sa_nn2")
	# Set up logging for predictions
	tensors_to_logNN2 = {"probabilities": "softmax_tensor"}
	logging_hook_nn2 = tf.train.LoggingTensorHook(tensors=tensors_to_logNN2, every_n_iter=50)

	r_sum = np.zeros(1)
	for i in range(1):
	    # Make the "Free-For-All" environment using the agent list
	    env.reset()
	    # Run the episodes just like OpenAI Gym
	    for i_episode in range(1):
	        state = env.reset()
	        
	        done = False
	        curr_state = None
	        prev_state = None
	        graph = np.random.rand(4, 30).astype("float32") + 0.0001
	#         print(graph)
	        pr_action = None
	        pr_pr_action = None
	        
	        while not done:
	#             env.render()
	            actions = env.act(state)
	            state, reward, done, info = env.step(actions)
	            r_sum[i] += reward[0]
	            
	            # as basic implementation I consider only one agent
	            prev_state = curr_state
	            curr_state = state

	            if pr_pr_action is not None:
	                # Train the model
	                for agent_num in range(4):
	                    train_input_NN2 = tf.estimator.inputs.numpy_input_fn(
	                        x={"state1": np.resize(
	                                state_to_matrix_with_action(curr_state[agent_num], action=pr_action[agent_num])\
	                                .astype("float32"), (1, 49*11)),
	                           "graph": np.resize(graph, (1, 4*30)) 
	                          },
	                        y=np.asarray([actions[agent_num]]),
	                        batch_size=1,
	                        num_epochs=None,
	                        shuffle=True)

	                    train_input_NN1 = tf.estimator.inputs.numpy_input_fn(
	                        x={"state1": np.resize(
	                                state_to_matrix_with_action(prev_state[agent_num], action=pr_pr_action[agent_num])\
	                                .astype("float32"), (1, 49 * 11)),
	                           "state2": np.resize(
	                                state_to_matrix_with_action(curr_state[agent_num], action=pr_action[agent_num])\
	                               .astype("float32"), (1, 49 * 11))},
	                        y=np.asmatrix(graph.flatten()),
	                        batch_size=1,
	                        num_epochs=None,
	                        shuffle=True)

	                    estimator_nn1.train(
	                        input_fn=train_input_NN1,
	                        steps=200,
	                        hooks=[logging_hook_nn1])

	                    estimator_nn2.train(
	                        input_fn=train_input_NN2,
	                        steps=200,
	                        hooks=[logging_hook_nn2])
	                    predictions = estimator_nn2.predict(input_fn=train_input_NN2)
	                    #next_action = np.array(list(p['classes'] for p in predictions))
	            pr_pr_action = pr_action
	            pr_action = actions
	        if i_episode > 300:
	            break
	    print('Game {} finished'.format(i))

	np.savetxt('result_2simple_2random.csv', r_sum, fmt='%1.4e')
	env.close()
	

if __name__ == '__main__':
	main()
