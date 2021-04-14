from ColoursAndData import *


class GameMode:

    @staticmethod
    def dis_to_food(arr, food):
        arr, food = np.array(arr), np.array(food)
        return np.linalg.norm(arr - food, axis=1)

    def find_directions(self, snake, food):
        directions = snake.available_steps()
        coor = np.array([np.array(snake.body[-1]) + np.array(DIRECTION_TO_TUPLE[i]) for i in directions])
        distances = self.dis_to_food(coor, food)
        return [x for _, x in sorted(zip(distances, directions), key=lambda pair: pair[0])]

    @staticmethod
    def printBlackTrial(c_snake, display):

        if WITH_GUI and display:
            hx, hy = c_snake.body[-1]
            pygame.draw.circle(display, BLACK, [hx, hy], SNAKE_RADIUS)
            pygame.display.update()
