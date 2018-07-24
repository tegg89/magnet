from enum import Enum

import numpy as np


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

    m = np.max(
        [my_position.shape[1], bomb_life.shape[1], board.shape[1], bombs.shape[1], can_kick.shape[1], ammo.shape[1],
         blast_strength.shape[1]])

    my_position1 = np.concatenate((my_position, np.zeros((my_position.shape[0], m - my_position.shape[1]))), axis=1)
    bomb_life1 = np.concatenate((bomb_life, np.zeros((bomb_life.shape[0], m - bomb_life.shape[1]))), axis=1)
    board1 = np.concatenate((board, np.zeros((board.shape[0], m - board.shape[1]))), axis=1)
    bombs1 = np.concatenate((bombs, np.zeros((bombs.shape[0], m - bombs.shape[1]))), axis=1)
    # enemies1 = np.concatenate((enemies, np.zeros((enemies.shape[0], m - enemies.shape[1]))), axis=1)
    can_kick1 = np.concatenate((can_kick, np.zeros((can_kick.shape[0], m - can_kick.shape[1]))), axis=1)
    ammo1 = np.concatenate((ammo, np.zeros((ammo.shape[0], m - ammo.shape[1]))), axis=1)
    blast_strength1 = np.concatenate((blast_strength, np.zeros((blast_strength.shape[0], m - blast_strength.shape[1]))),
                                     axis=1)

    result = np.concatenate((my_position1, bomb_life1, board1, board1, bombs1, can_kick1, ammo1, blast_strength1),
                            axis=0)
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

    m = np.max(
        [my_position.shape[1], bomb_life.shape[1], board.shape[1], bombs.shape[1], can_kick.shape[1], ammo.shape[1],
         blast_strength.shape[1]])

    my_position1 = np.concatenate((my_position, np.zeros((my_position.shape[0], m - my_position.shape[1]))), axis=1)
    bomb_life1 = np.concatenate((bomb_life, np.zeros((bomb_life.shape[0], m - bomb_life.shape[1]))), axis=1)
    board1 = np.concatenate((board, np.zeros((board.shape[0], m - board.shape[1]))), axis=1)
    bombs1 = np.concatenate((bombs, np.zeros((bombs.shape[0], m - bombs.shape[1]))), axis=1)
    # enemies1 = np.concatenate((enemies, np.zeros((enemies.shape[0], m - enemies.shape[1]))), axis=1)
    can_kick1 = np.concatenate((can_kick, np.zeros((can_kick.shape[0], m - can_kick.shape[1]))), axis=1)
    ammo1 = np.concatenate((ammo, np.zeros((ammo.shape[0], m - ammo.shape[1]))), axis=1)
    blast_strength1 = np.concatenate((blast_strength, np.zeros((blast_strength.shape[0], m - blast_strength.shape[1]))),
                                     axis=1)
    action = np.asmatrix(action)
    action1 = np.concatenate((action, np.zeros((action.shape[0], m - action.shape[1]))), axis=1)
    #     print(my_position1.shape, bomb_life1.shape, board1.shape, bombs1.shape, can_kick1.shape, ammo1.shape,
    #           blast_strength1.shape, action.shape, action1.shape)
    result = np.concatenate((my_position1, bomb_life1, board1, bombs1, can_kick1, ammo1, blast_strength1, action1),
                            axis=0)

    return np.asmatrix(result).flatten()
