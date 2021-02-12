from ColoursAndData import *
from GameMode import GameMode
from DFS import DFS


class A_STAR(GameMode):

    def __init__(self):
        self.dfs = DFS()

    def heu(self, snake, food):
        """
        Two option for the algorithm heuristic: Mannhatan distance or the Euclidean distance.
        """
        # res = snake.manhatan_distance_in_steps(food)
        res = snake.euclidean_dis(food)
        return res

    def dead_end(self, snake):
        """
        Upon the eating of the apple, the snake could get itself into an area bounded by it's own body - A dead end.
        Therefore, the snake would determine how risky would it be to eat the apple with the current body layout.
        If it has decided it is a dead end, it would sum the distances from the nearest body cells at each direction,
        and then push it to the end of the queue. Eventually, in there was no free way to retrieve the apple, the snake
        would choose the path that leads to such a layout which has the biggest sum of distances from the body, among
        found dead ends.
        """
        truth_table, total_dis = [], 0
        directions = self.available_steps(snake.curr_dir)
        for i, d in enumerate(directions):
            dir_t = DIRECTION_TO_TUPLE[d]
            s = snake.copy()
            # In each direction, go DEAD_END_RADIUS steps to determine if it is blocked.
            for j in range(DEAD_END_RADIUS):
                s.update_heuristic(dir_t)
                if s.fails():
                    truth_table.append(True)
                    total_dis += j + 1   # J starts from 0, so the real distance is bigger by 1.
                    break

            # This condition means that the current direction didn't fail, and therefore it is not a dead end.
            if len(truth_table) < i + 1:
                return False, total_dis
        return True, total_dis

    def find_route(self, snake, food, moves, display=None):
        """
        The A* algorithm uses an ongoing self-sorting stack.
        For more information about how the algo works, check out: https://www.youtube.com/watch?v=ySN5Wnu88nE&t=535s.
        """
        scores, visited = deque([self.heu(snake, food)]), {str(snake.body[-1]): (snake, 0, moves)}
        dead_moves = []
        stack = deque([str(snake.body[-1])])
        dead_counter, board_size = 1, GAME_HEIGHT * GAME_HEIGHT
        while len(stack):
            # Each turn take the next best node in the graph, to expand and look for the best path to the goal.
            c_head = stack.popleft()
            c_snake, g, c_moves = visited[c_head]
            c_score = scores.popleft()
            directions = self.find_directions(c_snake, food)

            # Expand all the children of c_head, perform the algo logic, and insert them to the sorted stack, according
            # their grade, from the smallest to the greatest.
            for j, d in enumerate(directions):
                dir_t = DIRECTION_TO_TUPLE[d]
                sk = c_snake.copy()
                sk.curr_dir = d
                sk.update_heuristic(dir_t)

                # Noes with such score is a node that is considered as a "dead end" (see self.dead_end docstring).
                # The longest path cannot be longer the board_size.
                if c_score >= board_size:
                    return True, c_moves
                # Do not add nodes that kill the snake to the stacks.
                if sk.fails():
                    continue
                elif sk.has_eaten(food):
                    # Before declaring victory - would the snake have a high probability to die after eating the apple?
                    # If so, do not use this path just yet, and try to look for a better one.
                    isDeadEnd, dis_from_body = self.dead_end(sk)
                    if isDeadEnd:
                        value = board_size * 10 - dis_from_body
                        head_s = str(sk.body[-1]) + "_" + str(dead_counter)
                        i = bisect_right(scores, value)
                        scores.insert(i, value)
                        stack.insert(i, head_s)
                        visited[head_s] = (sk, value, c_moves + [d])
                        dead_moves = visited[head_s][2]
                        dead_counter += 1
                        break
                    return True, c_moves + [d]

                # A* algorithm logic:
                dis = 0.3                                   # the value of the step to the next tile on the screen.
                t = g + dis + self.heu(sk, food) * 2
                i = bisect_right(scores, t)  # Find the index to push the new node to.

                head_s = str(sk.body[-1])
                ins_to_stack = True          # Is this the best path to the current node?
                if str(sk.body[-1]) in visited:
                    ins_to_stack = False
                    if t < visited[head_s][1]:
                        j = stack.index(head_s)
                        del stack[j]
                        del scores[j]
                        ins_to_stack = True

                if ins_to_stack:
                    # If this is the first occurrence of the node, or a better path has been found.
                    visited[head_s] = (sk, g + dis, c_moves + [d])
                    scores.insert(i, t)
                    stack.insert(i, head_s)

                # This variable is used for the case the snake dies, and we want to print to the board his very last
                # path on this earth.
                dead_moves = visited[head_s][2]
        return False, dead_moves


