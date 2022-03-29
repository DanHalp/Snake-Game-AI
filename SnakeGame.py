import pygame
import A_STAR
import BFS
import Snake
import DFS
# import BFS
from ColoursAndData import *


class SnakeGame:

    def __init__(self):
        self.clock = pygame.time.Clock()
        self.score = 1


    @staticmethod
    def init_display():
        dis = pygame.display.set_mode(((GAME_WIDTH - 1) * SIZE_FACTOR, (GAME_HEIGHT - 1) * SIZE_FACTOR))
        dis.fill(WHITE)
        pygame.display.update()
        return dis

    def init_game(self):
        s = Snake.Snake()
        return s, self.generate_food_smarter(s), 1

    @staticmethod
    def angle_to_bin(angle):
        """Each pixel in an image is surrounded by a maximum of 8 pixels, so an angle from the
        range of [0, 360) is out in one of 8 bins, where each bin is of size of 360 / 8 = 45
        degrees. """
        return (angle // 45) % 8

    @staticmethod
    def find_angle(pt1, pt2):

        pt1, pt2 = np.array(pt1), np.array(pt2)
        diff = (pt2 - pt1).squeeze()
        radians = np.arctan2(diff[:, 1], diff[:, 0])
        deg = (180 * radians / np.pi)
        deg = deg if len(deg.shape) < 2 else deg.squeeze()
        return deg

    @staticmethod
    def normalize(x):
        mu = np.mean(x)
        var = np.var(x)
        eps = 1e-5
        return (x - mu) / (var + eps)


    @staticmethod
    def decoy():
        """
         For debugging purposes, one can set the starting position of the game.
        """
        s = Snake.Snake()
        s.body[0] = (5, 2)
        s.body_set = Counter(s.body)
        s.curr_dir = 2
        food = (1, 3)
        return s, food, 1

    def message(self, msg, display, color, font_size, x, y, clear=False):
        """
        Print a message to the board.
        """
        if clear:
            display.fill(WHITE)

        font_style = pygame.font.SysFont(None, font_size)
        mesg = font_style.render(msg, True, color)
        display.blit(mesg, [x, y])
        pygame.display.update()

        if clear:
            # Keep the message on the board for a relatively long time
            self.clock.tick(0.7)

    def exit(self, display, score):
        pygame.display.update()
        # self.message("Oh, Crap", display, RED, 15, GAME_WIDTH / 10, 4 * GAME_HEIGHT / 5)
        # self.message("Your score: %s" % score, display, RED, 20, GAME_WIDTH / 10, 7 * GAME_HEIGHT / 8)
        # self.clock.tick(0.4)
        print(score)

    def pygameMUST(self):
        """
        There is a bug in pygame, that will get the game stuck unless the following is done.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

    @staticmethod
    def fill_display(display, snake, food, score):
        display.fill(WHITE)
        snake.draw_body(display)
        pygame.draw.circle(display, RED, np.array(food) * SIZE_FACTOR, SNAKE_RADIUS)
        # SnakeGame.message(snake, "Your score: %s" % score, display, BLACK, 25, 30, 30, clear=False)
        pygame.display.update()

    @staticmethod
    def generate_food(snake):

        board = np.zeros((GAME_WIDTH, GAME_HEIGHT))
        board[GRID[0], GRID[1]] = 1
        body = np.array(snake.body).T
        board[body[0], body[1]] = 0
        relevant_places = np.array(np.nonzero(board))
        index = np.random.choice(np.array(len(relevant_places[0])))
        food = relevant_places[:, index]
        return food

    def generate_food_smarter(self, snake):

        # bod = np.array(snake.body)
        # arr = []
        # while True:
        #     x = int(np.ceil(np.random.randint(GAME_WIDTH * 0.1, GAME_WIDTH * 0.9) / 10) * 10)
        #     y = int(np.ceil(np.random.randint(GAME_HEIGHT * 0.1, GAME_HEIGHT * 0.9) / 10) * 10)
        #     food = (x, y)
        #     if snake.body_set[food] == 0:
        #         arr.append(food)
        #     if len(arr) == 7:
        #         break
        # arr = np.array(arr)
        # baboon = np.zeros((arr.shape[0], bod.shape[0]))
        # for i, x in enumerate(arr):
        #     baboon[i] = np.linalg.norm(x - bod, axis=1)
        # dis = np.sum(baboon, axis=1)
        # food = tuple(arr[np.argmax(dis)])
        # return food
        pass

    def update_board(self, snake, food, direction_t):
        old_snake = snake.copy()
        snake.update_body(direction_t)
        hasEaten = snake.has_eaten(food, snake.body[-1])
        if hasEaten:
            fx, fy = old_snake.body[0]
            snake.add_tail((fx, fy))
            # food = self.generate_food_smarter(snake)
            food = self.generate_food(snake)

        return hasEaten, food

    def run_game(self, game_mode):

        if game_mode == PLAYER_MODE:
            self.player_run()

        elif game_mode == NN_MODE:
            self.nn_run()

        elif game_mode == SEARCH_MODE:
            self.tree_call()

        elif game_mode == BFS_MODE:
            self.dfs_bfs_game(game_mode=BFS_MODE)

        elif game_mode == DFS_MODE:
            self.dfs_bfs_game(game_mode=DFS_MODE)

        elif game_mode == A_STAR_MODE:
            self.a_star_game(game_mode=game_mode)

    def find_game_mode(self, game_mode):

        if game_mode == DFS_MODE:
            return DFS.DFS()
        elif game_mode == A_STAR_MODE:
            return A_STAR.A_STAR()
        elif game_mode == BFS_MODE:
            return BFS.BFS()
        raise Exception("Attention: Only DFS and A-STAR modes are available currently.")

    def a_star_game(self, game_mode=A_STAR_MODE, number_of_games=10):

        for _ in range(number_of_games):
            curr_snake, curr_food, score = self.decoy()
            mode_obj = self.find_game_mode(game_mode)

            display = None
            if WITH_GUI:
                display = self.init_display()
                self.fill_display(display, curr_snake, curr_food, score)

            while True:
                s = curr_snake.copy()
                success, moves = mode_obj.find_route(s, curr_food, [], display=display)

                for i, move in enumerate(moves):
                    d = DIRECTION_TO_TUPLE[move]
                    b, curr_food = self.update_board(curr_snake, curr_food, d)

                    if WITH_GUI:
                        self.fill_display(display, curr_snake, curr_food, score)

                    self.pygameMUST()
                    self.clock.tick(20)

                if not success or not len(moves):

                    if WITH_GUI:
                        self.exit(display, score)
                    break
                score += 1
                curr_snake.curr_dir = moves[-1]

    def dfs_bfs_game(self, game_mode=DFS_MODE, number_of_games=10):
        for _ in range(number_of_games):

            curr_snake, curr_food, score = self.decoy()
            mode_obj = self.find_game_mode(game_mode)

            display = None
            if WITH_GUI:
                display = self.init_display()
                self.fill_display(display, curr_snake, curr_food, score)

            while True:
                s = curr_snake.copy()
                success, moves = mode_obj.find_route(s, curr_food, [], display if SHOW_POLICY else None)
                for i, move in enumerate(moves):
                    d = DIRECTION_TO_TUPLE[move]
                    b, curr_food = self.update_board(curr_snake, curr_food, d)
                    if WITH_GUI:
                        self.fill_display(display, curr_snake, curr_food, score)
                    self.pygameMUST()
                    self.clock.tick(30)

                if not success or not len(moves):
                    if WITH_GUI:
                        self.exit(display, score)
                    break
                score += 1
                curr_snake.curr_dir = moves[-1]



def main():
    pygame.init()
    game = SnakeGame()
    game.run_game(game_mode=BFS_MODE)
    pygame.quit()

#
if __name__ == '__main__':
    main()
