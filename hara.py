import multiprocessing
import pickle
import random
from copy import deepcopy

from Snake import Snake
from ColoursAndData import *
from GameMode import GameMode
from SnakeGame import SnakeGame


class NNSnake(Snake):

    def __init__(self, init_mode=DECOY):
        super().__init__()
        self.food = None
        self.set_body_and_food(init_mode)
        self.ws = [np.zeros((INPUT_SIZE, HIDDEN_SIZE_1)),
                   np.zeros((HIDDEN_SIZE_1, HIDDEN_SIZE_2)),
                   np.zeros((HIDDEN_SIZE_2, OUTPUT_SIZE))]
        self.bs = [np.zeros(HIDDEN_SIZE_1),
                   np.zeros(HIDDEN_SIZE_2),
                   np.zeros(OUTPUT_SIZE)]
        self.score = 0
        self.total_steps = 0
        self.round = 0

    def set_body_and_food(self, reset_mode=DECOY):
        food, head = NN.decoy()
        self.body = deque([head])
        self.body_set = Counter()
        self.body_set[head] += 1
        self.curr_dir = 3 # np.random.choice(np.arange(4))
        self.food = food

    def calc_dis_snake_food(self):

        vec1 = np.array(self.body[-1])
        vec2 = np.array(self.food)
        l = np.linalg.norm(vec1 - vec2)
        return l / 10

    def degree_from_food(self):
        vec1 = np.array(DIRECTION_TO_TUPLE[self.curr_dir])
        vec2 = np.array(self.food) - np.array(self.body[-1])
        angle = (np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])) * 180 / np.pi
        return angle / 30

    def dis_to_threats(self):
        return [self.dis_from_threat(d, self.food) for d in self.available_steps()]

    def create_input(self):
        return np.array([1 / self.calc_dis_snake_food(),
                         self.degree_from_food(),
                         *self.dis_to_threats()])

    def forward(self):
        directions = self.available_steps()
        current_layer = self.create_input()
        for i in range(len(self.ws)):
            current_layer = NN.activation(current_layer.dot(self.ws[i]) + self.bs[i])
        current_layer = NN.put_in_range(current_layer, 0, 1)
        current_layer = NN.activation(current_layer, SIGMOID)
        max_val = np.max(current_layer)
        indices = current_layer == max_val
        optional_dir = directions[indices]
        return np.random.choice(optional_dir)

    def reward(self):
        return self.score  #- self.total_steps * 0.5

    def update(self, d):

        eaten, failed = False, False
        xc, yc = d
        hx, hy = self.body[-1]
        new_head = (xc + hx, hy + yc)

        if self.fails(new_head):
            failed = True
        elif self.has_eaten(self.food, new_head):
            self.add_cell(new_head)
            self.food = NN.generate_food(self)
            eaten = True
        else:
            to_remove = self.body[0]
            self.body_set[to_remove] = 0
            self.body_set[new_head] += 1
            self.body.rotate(-1)
            self.body[-1] = new_head

        return eaten, failed

    def run(self):

        self.score = 0
        self.total_steps = 0
        scores = []

        score = 0
        eaten, failed = False, False
        steps = 0
        self.set_body_and_food()
        while True:
            if steps >= MAX_STEPS or failed:
                break
            move = self.forward()
            self.curr_dir = move
            d = DIRECTION_TO_TUPLE[move]
            eaten, failed = self.update(d)
            if eaten:
                score += 1
                steps = -1
            steps += 1
            self.total_steps += 1
        return score

    def copy(self):
        s = NNSnake(self.food)
        s.body = deque(self.body.copy())
        s.body_set = self.body_set.copy()
        s.curr_dir = self.curr_dir
        s.depth = self.depth
        return s


class NN(GameMode, SnakeGame):

    def __init__(self, n_threads=10):
        super().__init__()
        self.generation = [NNSnake() for _ in range(GEN_SIZE)]
        self.n_threads = n_threads
        self.best_candidate = self.generation[0]
        self.data = []

    @staticmethod
    def decoy():
        food = (20, 30)
        head = (20, 20)
        return food, head

    def mate(self):

        def roulette_wheel(curr_pop):
            new_pop = []
            wheel_sum = np.sum([s.reward() for s in curr_pop])
            for _ in range(GEN_SIZE):
                pick = np.random.uniform(0, wheel_sum)
                current_sum = 0
                for cand in curr_pop:
                    current_sum += cand.reward()
                    if current_sum >= pick:
                        new_pop.append(deepcopy(cand))
                        break
            return new_pop

        def get_offsprings(x: NNSnake, y: NNSnake, par_chance=0.7):

            for i in range(len(x.ws)):
                w_change, b_change = np.random.rand(*x.ws[i].shape) < par_chance, np.random.rand(*x.bs[i].shape) < par_chance

                x.ws[i][~w_change] = y.ws[i][~w_change]
                y.ws[i][w_change] = x.ws[i][w_change]
                x.bs[i][~b_change] = y.bs[i][~b_change]
                y.bs[i][b_change] = x.bs[i][b_change]

            return [x, y]

        new_pop = roulette_wheel(self.generation)

        self.generation += new_pop
        n_children = GEN_SIZE
        random.shuffle(self.generation)
        for i in range(0, n_children, 2):
            pair = self.generation[i: i + 2]
            offsprings = get_offsprings(*sorted(pair, key=lambda s: s.reward()))
            self.generation += offsprings

    def mutate(self):

        for s in self.generation:
            for i in range(len(s.ws)):
                h, w = s.ws[i].shape
                indices = np.random.rand(h, w) < MUTATION_THRESH
                gauss_values = np.random.normal(size=(h, w))
                s.ws[i][indices] += gauss_values[indices]

                h = len(s.bs[i])
                indices = np.random.rand(h) < MUTATION_THRESH
                gauss_values = np.random.normal(size=h)
                s.bs[i][indices] += gauss_values[indices]

    @staticmethod
    def function_f(s):
        return s.run()

    def evaluate(self, pool=None):
        [s.set_body_and_food() for s in self.generation]
        scores = np.array(pool.map(NN.function_f, self.generation))


        for i in range(len(scores)):
            self.generation[i].score = scores[i]

        self.generation = list(sorted(self.generation, key=lambda a: a.reward(), reverse=True))[:GEN_SIZE]
        curr_best_snake_avg = self.generation[0].score
        curr_gen_avg = np.mean(scores)
        print("{:.2f}".format(curr_best_snake_avg), "{:.2f}".format(curr_gen_avg))
        self.data.append((curr_best_snake_avg, curr_gen_avg))

    def train(self, number_of_rounds=60):

        pool = multiprocessing.Pool(self.n_threads)
        for j in range(number_of_rounds):
            self.mate()
            self.mutate()
            print(f"Iter {j}: ", end="")
            self.evaluate(pool)

        self.best_candidate = self.generation[0]
        with open("filename.pickle", "wb") as file:
            pickle.dump(self.best_candidate, file)

    def play_with_winner(self, number_of_games=5):

        for _ in range(number_of_games):
            steps = 0
            score = 0
            curr_snake = self.best_candidate
            curr_snake.set_body_and_food()

            display = None
            if WITH_GUI:
                display = self.init_display()
                self.fill_display(display, curr_snake, curr_snake.food, score)

            while True:
                move = curr_snake.forward()
                curr_snake.curr_dir = move
                d = DIRECTION_TO_TUPLE[move]
                eat, fails = curr_snake.update(d)
                if WITH_GUI:
                    self.fill_display(display, curr_snake, curr_snake.food, score)

                if not eat:
                    steps += 1
                else:
                    score += 1
                    steps = 0

                self.pygameMUST()
                self.clock.tick(150)

                if fails or steps > MAX_STEPS:
                    if WITH_GUI:
                        self.exit(display, score)
                    break


    @staticmethod
    def activation(x, activation=RELU):
        assert activation in ACTIVATIONS
        return ACTIVATIONS[activation](x)

    @staticmethod
    def put_in_range(arr, newMin, newMax):
        oldMax, oldMin = np.max(arr), np.min(arr)
        oldRange = oldMax - oldMin
        NewRange = newMax - newMin
        return (arr - oldMin) * NewRange / oldRange + newMin if oldRange else arr


if __name__ == '__main__':
    x = 100
    y = 100
    import os

    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)

    # import pygame

    pygame.init()
    screen = pygame.display.set_mode((100, 100))

    # pygame.init()
    nn = NN()
    # nn.train(number_of_rounds=70)
    load_cand = 0
    save_cand = 1
    if save_cand:
        nn.train(number_of_rounds=NUM_TRAINING_ROUNDS)
        with open("data.pickle", "wb") as file:
            pickle.dump(nn.data, file)

        with open("filename.pickle", "wb") as file:
            pickle.dump(nn.best_candidate, file)


    if load_cand:

        with open("data.pickle", "rb") as data_file:
            data = np.array(pickle.load(data_file))
            titles = ["best_cand_score", "gen_avg"]
            for i in range(data.shape[1]):
                plt.plot(np.arange(NUM_TRAINING_ROUNDS), data[:, i], label=titles[i])

            plt.legend()
            plt.show()

        with open("filename.pickle", "rb") as file:
            nn.best_candidate = pickle.load(file)
            nn.play_with_winner(number_of_games=100)

    pygame.quit()













