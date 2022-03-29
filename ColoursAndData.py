import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import pygame
from collections import Counter
from collections import deque
from bisect import insort_left, insort_right, bisect_right
from scipy.stats import truncnorm
from copy import deepcopy
from sklearn import preprocessing

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PINK = (255, 200, 200)
DARK_BLUE = (0, 0, 128)
YELLOW = np.array((0, 255, 255))


# SNAKE GAME MAGIC NUMBERS:
BORDER = 2
W, H = 10, 10
GAME_WIDTH = W + BORDER
GAME_HEIGHT = H + BORDER
MAXIMUM_SCORE = W * H - 1
SIZE_FACTOR = 50
BOARD_SIZE = GAME_HEIGHT * GAME_WIDTH
SNAKE_RADIUS = SIZE_FACTOR // 2

GRID = np.array(np.meshgrid(np.arange(1, GAME_WIDTH - 1), np.arange(1, GAME_HEIGHT - 1))).reshape(2, -1)


PLAYER_MODE = 0
NN_MODE = 1
SEARCH_MODE = 2
BFS_MODE = 3
DFS_MODE = 4
A_STAR_MODE = 5
CLOCK_TICK = 1000
WITH_GUI = True
PI = np.pi
PI2 = 2 * PI


LEFT_ARROW = 0
UP_ARROW = 1
RIGHT_ARROW = 2
DOWN_ARROW = 3

# NN
POOL = True
LEFT = -1
FRONT = 0
RIGHT = 1
NN_NUMBER_OF_GAMES = 75
DIRECTIONS = [LEFT, FRONT, RIGHT]
INPUT_SIZE = 15
HIDDEN_SIZE_1 = 20 # 10
HIDDEN_SIZE_2 = 15 # 10
OUTPUT_SIZE = 3
NUM_LAYERS = 3
save_cand = 3
SHOW_POLICY = True
GEN_SIZE = 75
MUTATION_THRESH = 0.03
FINAL_MUTATION_RATE = 0.03
MAX_STEPS = BOARD_SIZE
CROSS_VALUE = max(W, H)
SIGMOID = 0
SIGMOID_NEW = 2
TANH = 3
SIN = 4
RELU = 1
DECOY = 0
RAND = 1
N_THREADS = 10
NUM_TRAINING_ROUNDS = 1500
NUMBER_OF_TRIES = 2000
ACTIVATIONS = {RELU: lambda x: np.maximum(0, x), SIGMOID: lambda x: 1 / (1 + np.exp(-x)),
               SIGMOID_NEW: lambda x: 1 / (1 + np.exp(3 - x)), TANH: lambda x: np.tanh(2*x),
               SIN: lambda x: 1.2 * np.sin(x)}
NORMALIZE = lambda x: x / CROSS_VALUE
MUTATION_F = 0.05

#ADAM
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99

# Tree call algorithm
MAX_VALUE = max(GAME_WIDTH, GAME_HEIGHT)
SEARCH_DEPTH = int(MAX_VALUE / 10)
SURROUNDINGS = 9 * MAX_VALUE / 100

# A*
DEAD_END_RADIUS = 10

border_punishments = [150 * SEARCH_DEPTH] + [40 * i for i in range(SEARCH_DEPTH-1, -1, -1)]
DIRECTION_TO_TUPLE = np.array([(-1, 0), (0, -1), (1, 0), (0, 1)])
DIRECTION_TO_ANGLE = np.array([np.pi, np.pi * 0.5, 0, 1.5 * np.pi])
# ROUND_DIRECTIONS =

f_counter = 0
files = glob.glob("./**/*.jpg")
if len(files):
    for f in files:
        x = max([int(n) for n in re.findall(r"\d+", f)])
        if x >= f_counter:
            f_counter = x + 1