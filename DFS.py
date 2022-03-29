from ColoursAndData import *
from GameMode import GameMode

def set_final_moves(final_moves, t_moves):
    return t_moves if len(t_moves) >= len(final_moves) else final_moves


class DFS(GameMode):
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
        stack = deque([str(snake.body[-1])])
        visited = {str(snake.body[-1]):  (snake, moves)}
        final_moves = []

        while stack:
            c_head = stack.pop()
            c_snake, c_moves = visited[c_head]
            directions = list(reversed(self.find_directions(c_snake, food)))

            for i, d in enumerate(directions):
                t_snake = c_snake.copy()
                dir_t = DIRECTION_TO_TUPLE[d]
                t_snake.curr_dir = d
                t_moves = c_moves.copy() + [d]
                t_snake.update_heuristic(dir_t)
                t_head = str(t_snake.body[-1])
                if t_head not in visited:

                    if t_snake.fails():
                        final_moves = set_final_moves(final_moves, t_moves)
                        continue

                    elif t_snake.has_eaten(food, t_snake.body[-1]):
                        return True, t_moves

                    stack.append(t_head)
                    visited[t_head] = (t_snake, t_moves)
                    final_moves = set_final_moves(final_moves, t_moves)

        return False, final_moves

