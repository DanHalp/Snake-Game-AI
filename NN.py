import numpy as np
from ColoursAndData import *

class NN:

    def __init__(self, snake):
        self.snake = snake
        self.food = None
        self.x_data = []
        self.y_data = []

    def calc_dis_snake_food(self, p1):

        vec1 = np.array(p1)
        vec2 = np.array(self.food)
        l = np.linalg.norm(vec1 - vec2)
        return l


    def create_training_data(self):
        pass
#
#
# def angle_between(p1, p2):
#     ang1 = np.arctan2(*p1[::-1])
#     ang2 = np.arctan2(*p2[::-1])
#     return np.rad2deg((ang1 - ang2) % (2 * np.pi))
#
# vec0 = np.array([1,0])
# vec1 = np.array([1, 0])
# vec2 = np.array([-1, -1])
#
# deg1 = angle_between(vec1, vec0)
# deg2 = angle_between(vec2, vec0)
#
# print( deg1, deg2, deg1 - deg2 )
