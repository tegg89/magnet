import numpy as np

REWARD_FOR_KILLING = 100
REWARD_FOR_KILLING_YOURSELF = -100
REWARD_FOR_INCREASING_CAN_KICK = 10
REWARD_FOR_INCREASING_AMMO = 10
REWARD_FOR_INCREASING_BLAST_POWER = 10
REWARD_FOR_DESTROING_WOODEN_WALL = 20

# contain dictionary which describe current bombs on field
list_of_sest_boobms = [[], [], [], []]
# contain dictionary of dead agent, coordinates of bombs which kill them and flag 'was dead on privius step'
killed = {}
# dictionary of coordinate_of_adgent
coordinate_of_adgent = {0: [0, 0],
                        1: [0, 0],
                        2: [0, 0],
                        3: [0, 0]}


def init_list_of_vertex(board):
    vertex_name = {}  # dictionary of coordinate (converted to x*11 + y, except agents: they alwas stay in 121, 122,
    # 123 and 124  + converted position) to type of vertex
    vertex_list = []  # list of existed vertex in current state
    for i in range(11):
        for j in range(11):
            if board[i, j] == 2:
                vertex_name[i * 11 + j] = 'wooden wall'
                vertex_list.append(i * 11 + j)
            if board[i, j] == 3:
                vertex_name[i * 11 + j] = 'bomb'
                vertex_list.append(i * 11 + j)
            if board[i, j] == 4:
                vertex_name[i * 11 + j] = 'flames'
                vertex_list.append(i * 11 + j)
            if board[i, j] == 5:
                vertex_name[i * 11 + j] = 'fog'
                vertex_list.append(i * 11 + j)
            if board[i, j] == 6:
                vertex_name[i * 11 + j] = 'extra bomb'
                vertex_list.append(i * 11 + j)
            if board[i, j] == 7:
                vertex_name[i * 11 + j] = 'increase range'
                vertex_list.append(i * 11 + j)
            if board[i, j] == 8:
                vertex_name[i * 11 + j] = 'can kick Power-Up'
                vertex_list.append(i * 11 + j)
            if board[i, j] == 10:
                vertex_name[121] = 0
                vertex_list.append(121)
            if board[i, j] == 11:
                vertex_name[122] = 1
                vertex_list.append(122)
            if board[i, j] == 12:
                vertex_name[123] = 2
                vertex_list.append(123)
            if board[i, j] == 13:
                vertex_name[124] = 3
                vertex_list.append(124)
    return vertex_name, vertex_list


def check_next_to_bomb(graph, agent_num, current_state, privius_vertex_name, reward):
    print(agent_num)
    agent_num = int(agent_num)
    for bomb in list_of_sest_boobms[agent_num]:
        bomb['blast strength'] = current_state[bomb['x'] + 23, bomb['y']]
        bomb['life'] = current_state[bomb['x'] + 1, bomb['y']]
        if (int(bomb['life']) == 1):
            # booom
            for key, val in privius_vertex_name.items():
                item_x = key // 11
                item_y = key % 11
                # if it is an agent -- watch coordinet in coordinate_of_adgent
                if val in [0, 1, 2, 3]:
                    item_x, item_y = coordinate_of_adgent[val]

                # check is this item will be killed by bomb
                if ((bomb['x'] - bomb['blast strength'] < item_x < bomb['x'] + bomb['blast strength']) and (
                        item_y == bomb['y'])) or \
                        ((bomb['y'] - bomb['blast strength'] < item_y < bomb['y'] + bomb['blast strength']) and (
                                item_x == bomb['x'])):

                    # kill someone
                    if val in [0, 1, 2, 3] and val != agent_num:
                        print(agent_num, " kill ", val)
                        killed[val] = {'x': bomb['x'],
                                       'y': bomb['y'],
                                       'was dead on previous step': False
                                       }
                        reward += REWARD_FOR_KILLING
                        # learn that agent, which was killed should avoid this bomb
                        graph[val, int(bomb['x']) * 11 + int(bomb['y'])] = -100

                    # kill yourself
                    if val == agent_num:
                        print(agent_num, " kill itself")
                        killed[val] = {'x': bomb['x'],
                                       'y': bomb['y'],
                                       'was dead on previous step': False
                                       }
                        reward += REWARD_FOR_KILLING_YOURSELF

                        # learn that agent, which was killed should avoid this bomb
                        graph[val, int(bomb['x']) * 11 + int(bomb['y'])] = -100

                    # destroy wooden wall
                    if val == 'wooden wall':
                        print(agent_num, " destroyed wooden wall")
                        reward += REWARD_FOR_DESTROING_WOODEN_WALL
            # delete bomb after booom
            list_of_sest_boobms[agent_num].remove(bomb)

    return graph, reward


def reward_shaping(graph, curr_state, prev_state, agent_num):
    coordinate_of_adgent[agent_num] = (int(curr_state[0, 0]), int(curr_state[0, 1]))

    prev_state = np.asmatrix(prev_state).reshape(38, 11)
    curr_state = np.asmatrix(curr_state).reshape(38, 11)

    curr_vertex_name, curr_vertex_list = init_list_of_vertex(curr_state[12:23])
    prev_vertex_name, prev_vertex_list = init_list_of_vertex(prev_state[12:23])

    prev_x = int(prev_state[0, 0])
    prev_y = int(prev_state[0, 1])

    reward = 0

    # on privius state agent lay bomb
    if prev_state[37, 0] == 5:
        for key, val in prev_vertex_name.items():
            if val == 'bomb':  # consider all setted bombs
                bomb_x = key // 11
                bomb_y = key % 11
                # check if that bomb was next to adgent in privius state
                if abs(bomb_x - prev_x) + abs(bomb_y - prev_y) == 1:
                    list_of_sest_boobms[agent_num].append({'x': bomb_x,
                                                           'y': bomb_y,
                                                           'life': curr_state[bomb_x + 1, bomb_y],
                                                           'blast strength': curr_state[bomb_x + 23, bomb_y]})
                    # add list of [x, y, bomb life, boobm blast strength]

    # increase can kick
    if prev_state[34, 0] < curr_state[34, 0]:
        print(agent_num, " increase can kick")
        graph[agent_num, prev_x * 11 + prev_y] = 10  # set edge between can kick power up and adjent as 10
        reward += REWARD_FOR_INCREASING_CAN_KICK

    # increase ammo
    if prev_state[35, 0] < curr_state[35, 0]:
        print(agent_num, " increase ammo")
        graph[agent_num, (prev_x * 11 + prev_y) % 120] = 10  # set edge between increase ammo power up and adjent as 10
        reward += REWARD_FOR_INCREASING_AMMO

    # increase blast power
    if prev_state[36, 0] < curr_state[36, 0]:
        print(agent_num, " increase blast power")
        graph[agent_num, prev_x * 11 + prev_y] = 10  # set edge between blast power power up and adjent as 10
        reward += REWARD_FOR_INCREASING_BLAST_POWER

    graph, reward = check_next_to_bomb(graph, agent_num, curr_state, prev_vertex_name, reward)

    # has died
    if agent_num in killed and not killed[agent_num]['was dead on previous step']:
        print(agent_num, " has dead")
        killed[agent_num]['was dead on previous step'] = 'True'
        return graph

    return graph.astype("float32"), reward
