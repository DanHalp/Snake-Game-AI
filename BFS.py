import numpy as np

from ColoursAndData import *
from GameMode import GameMode

def set_final_moves(final_moves, t_moves):
    return t_moves if len(t_moves) >= len(final_moves) else final_moves



def fill_display(display, cell, food, color):
    # display.fill(WHITE)
    pygame.draw.circle(display, RED, np.array(food) * SIZE_FACTOR, SNAKE_RADIUS)
    pygame.draw.circle(display, color, np.array(cell) * SIZE_FACTOR, SNAKE_RADIUS)
    # SnakeGame.message(snake, "Your score: %s" % score, display, BLACK, 25, 30, 30, clear=False)
    pygame.display.update()

def fill_display2(display, snake, food):
    display.fill(WHITE)
    snake.draw_body(display)
    pygame.draw.circle(display, RED, np.array(food) * SIZE_FACTOR, SNAKE_RADIUS)
    pygame.display.update()

def my_sleep(num, display=None):
    for _ in range(num):
        if display:
            pygame.event.get()

class BFS(GameMode):
    """
    Solve one Snake Game with the DFS algorithm.
    """

    # Main functionality
    def __init__(self):
        self.route_success = False
        self.final_moves = []

    def find_route(self, snake, food, moves, display=None):
        """
        Find a path to the apple, using DFS.
        Note that the children nodes are picked with preferences (distance to the apple), for efficiency reasons.
        """
        queue = deque([str(snake.body)])
        visited = {str(snake.body): (snake, moves)}
        final_moves, children = [], []
        color_counter = 0
        num_child, iteration = 3, 0
        colour = GREEN
        while queue:
            pygame.event.get()
            c_head = queue.popleft()
            c_snake, c_moves = visited[c_head]
            directions = list(reversed(self.find_directions(c_snake, food)))

            # if display is not None:
            #     if iteration == num_child:
            #         num_child *= 3
            #         iteration = 0
            #         color_counter = (color_counter + 1) % 2
            #
            #     colour = [GREEN, BLACK][color_counter]
            i = np.random.choice(np.arange(3))
            colour = [GREEN, BLACK, PINK][i]

            for i, d in enumerate(directions):
                iteration += 1
                t_snake = c_snake.copy()
                dir_t = DIRECTION_TO_TUPLE[d]
                t_snake.curr_dir = d
                t_moves = c_moves.copy() + [d]
                t_snake.update_heuristic(dir_t)
                t_body = str(t_snake.body)
                if t_body not in visited:
                    if t_snake.fails():
                        final_moves = set_final_moves(final_moves, t_moves)
                        continue

                    elif t_snake.has_eaten(food, t_snake.body[-1]):
                        return True, t_moves

                    queue.append(t_body)
                    visited[t_body] = (t_snake, t_moves)
                    final_moves = set_final_moves(final_moves, t_moves)

                    if snake.body_set[t_snake.body[-1]] == 0 and display is not None:
                        fill_display(display, t_snake.body[-1], food, colour)
                        # fill_display2(display, t_snake, food)
                        my_sleep(30000, display)

        return False, final_moves

