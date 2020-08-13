import numpy as np
import matplotlib.pyplot as plt
import pygame
from collections import Counter
from collections import deque
from bisect import insort_left, insort_right, bisect_right

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PINK = (255, 200, 200)
DARK_BLUE = (0, 0, 128)
YELLOW = np.array((0, 255, 255))
SNAKE_RADIUS = 5

# SNAKE GAME MAGIC NUMBERS:
GAME_WIDTH = 250
GAME_HEIGHT = 200
PLAYER_MODE = 0
NN_MODE = 1
SEARCH_MODE = 2
BFS_MODE = 3
DFS_MODE = 4
A_STAR_MODE = 5
CLOCK_TICK = 1000
WITH_GUI = True

LEFT_ARROW = 0
UP_ARROW = 1
RIGHT_ARROW = 2
DOWN_ARROW = 3

# NN
LEFT = -1
FRONT = 0
RIGHT = 1
NN_NUMBER_OF_GAMES = 40
DIRECTIONS = [LEFT, FRONT, RIGHT]

# Tree call algorithm
MAX_VALUE = max(GAME_WIDTH, GAME_HEIGHT)
SEARCH_DEPTH = int(MAX_VALUE / 10)
SURROUNDINGS = 9 * MAX_VALUE / 100

# A*
DEAD_END_RADIUS = 10

border_punishments = [150 * SEARCH_DEPTH] + [40 * i for i in range(SEARCH_DEPTH-1, -1, -1)]
DIRECTION_TO_TUPLE = {LEFT_ARROW: (-10, 0), RIGHT_ARROW: (10, 0), UP_ARROW: (0, -10), DOWN_ARROW: (0, 10), -1: (0, 0)}
