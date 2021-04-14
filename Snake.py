from ColoursAndData import *


class Snake:

    def __init__(self):

        self.body = deque([])
        self.body_set = Counter()
        self.curr_dir = np.random.choice(range(4))
        self.create_snake()

        # A star fields.

        self.heuristic_value = 0
        self.depth = 0

    def available_steps(self):
        return (self.curr_dir + np.array(DIRECTIONS)) % 4

    def create_snake(self):
        x = int(np.ceil(np.random.randint(GAME_WIDTH * 0.25, GAME_WIDTH * 0.75)))
        y = int(np.ceil(np.random.randint(GAME_HEIGHT * 0.25, GAME_HEIGHT * 0.75)))
        self.add_cell((x, y))

    def add_tail(self, tail):
        self.body.appendleft(tail)

    def add_cell(self, cell):
        assert self.body_set[cell] <= 1
        self.body.append(cell)
        self.body_set[cell] += 1

    def draw_body(self, display):
        for i, cell in enumerate(self.body):
            x, y = np.array(cell) * SIZE_FACTOR
            if i + 4 >= len(self.body) - 1:
                j = len(self.body) - i
                pygame.draw.circle(display, np.maximum(YELLOW - (20 * j), np.zeros(3)), [x, y], SNAKE_RADIUS)
            else:
                pygame.draw.circle(display, BLUE, [x, y], SNAKE_RADIUS)

    def update_body(self, direction):

        end_t = self.body[0]
        x_change, y_change = direction
        x, y = self.body[-1]
        self.body.rotate(-1)
        self.body[-1] = (x + x_change, y + y_change)
        self.body_set[self.body[-1]] += 1
        self.body_set[end_t] = 0

    def has_eaten(self, food, new_head):
        hx, hy = new_head
        fx, fy = food
        return hx == fx and hy == fy

    def copy(self):
        s = Snake()
        s.body = deque(self.body.copy())
        s.body_set = self.body_set.copy()
        s.curr_dir = self.curr_dir
        s.depth = self.depth
        return s

    def update_heuristic(self, change_values):

        xc, yc = change_values
        hx, hy = self.body[-1]
        new_head = (xc + hx, hy + yc)

        to_remove = self.body[0]

        self.body_set[to_remove] = 0
        self.body_set[new_head] += 1
        self.body.rotate(-1)
        self.body[-1] = new_head

    def hits_wall(self, x, y):
        return x <= 0 or x >= GAME_WIDTH or y <= 0 or y >= GAME_HEIGHT

    def bites_itself(self, new_head):
        return self.body_set[tuple(new_head)] > 1

    def fails(self, new_head):
        return self.hits_wall(*new_head) or self.bites_itself(new_head)

    def euclidean_dis(self, elememt):
        head = np.array(self.body[-1])
        food = np.array(elememt)
        return np.linalg.norm(head - food) / 10

    def dis_from_threat(self, direction, food):

        dummy = self.copy()
        c_dir = DIRECTION_TO_TUPLE[direction]
        new_head = np.array(dummy.body[-1]) + c_dir
        return 0 if self.fails(new_head) else 1

    def find_directions(self):
        # directions = snake.available_steps()
        # coor = np.array([np.array(snake.body[-1]) + np.array(DIRECTION_TO_TUPLE[i]) for i in directions])
        # distances = self.dis_to_food(coor, food)
        # return [x for _, x in sorted(zip(distances, directions), key=lambda pair: pair[0])]
        pass
