import multiprocessing
import pickle
import time
from copy import deepcopy

import cv2

from Snake import Snake
from ColoursAndData import *
from GameMode import GameMode
from SnakeGame import SnakeGame


class NNSnake(Snake):

    def __init__(self, init_mode=DECOY):
        super().__init__()
        self.food = None
        self.set_body_and_food(init_mode)
        self.ws = [np.random.normal(size=(INPUT_SIZE, HIDDEN_SIZE_1)),
                   np.random.normal(size=(HIDDEN_SIZE_1, HIDDEN_SIZE_2)),
                   np.random.normal(size=(HIDDEN_SIZE_2, OUTPUT_SIZE))]
        # self.ws = [np.random.uniform(size=(INPUT_SIZE, HIDDEN_SIZE_1)),
        #            np.random.normal(size=(HIDDEN_SIZE_1, HIDDEN_SIZE_2)),
        #            np.random.normal(size=(HIDDEN_SIZE_2, OUTPUT_SIZE))]
        self.bs = [np.zeros(HIDDEN_SIZE_1),
                   np.zeros(HIDDEN_SIZE_2),
                   np.zeros(OUTPUT_SIZE)]

        self.score = 0
        self.total_steps = 0
        self.death_penalty = 0
        self.rew = 0
        self.round = 0
        self.foods_so_far = []

    def set_body_and_food(self, reset_mode=DECOY):
        food, head = NN.decoy()
        self.body = deque([head])
        self.body_set = Counter()
        self.body_set[head] += 1
        self.curr_dir = 3
        self.food = food

    @staticmethod
    def get_delta(deg1, deg2):
        delta = (PI2 - deg1)
        X = ((deg2[:, None] + delta) % PI2).squeeze()
        ret = np.array(np.where(np.pi < X, 2 * np.pi - X, -X))
        return ret

    def deg_to_objects(self, source_ang, obj):

        obj = np.array(obj)
        obj = obj if len(obj.shape) > 1 else obj[None]
        source_ang = np.array([source_ang]) if type(obj) == int else source_ang
        obj_rel = (obj - np.array(self.body[-1])) * [1, -1]
        ang_obj = np.arctan2(obj_rel[:, 1], obj_rel[:, 0]) % PI2

        return NNSnake.get_delta(source_ang, ang_obj)

    def bins_of_objects(self, dirs, obj):

        dirs = np.array([dirs]) if type(dirs) == int else np.array(dirs)
        source_ang = DIRECTION_TO_ANGLE[dirs]
        angels = self.deg_to_objects(source_ang, obj)
        bins = np.abs(angels) <= PI * 0.25
        bins = bins[None] if len(bins.shape) < 2 else bins
        idx = np.any(bins, axis=1)
        ret = np.argmax(bins, axis=1)
        ret[np.logical_not(idx)] = -1
        return ret

    def dis_to_walls(self):
        boundaries = np.array([[0, 0], [0, 0], [GAME_WIDTH - 1, 0], [0, GAME_HEIGHT - 1]])
        availabe_s = self.available_steps()
        head = np.array(self.body[-1])
        moves = DIRECTION_TO_TUPLE[availabe_s]
        dis_to_walls = np.linalg.norm(boundaries[availabe_s] - np.multiply(moves, head), axis=1)
        return dis_to_walls

    def dis_to_food(self):
        posses = self.surroundings_pixels()
        ret = np.linalg.norm(posses - np.array(self.food), axis=1)
        # return ACTIVATIONS[SIN](ret)
        return ret

    def deg_to_food(self):
        food_deg = self.deg_to_objects(DIRECTION_TO_ANGLE[self.available_steps()], self.food)

        return food_deg

    def dis_to_body(self):
        if len(self.body) < 3:
            return np.zeros(3)

        body = np.array(self.body)
        dirs = self.available_steps()
        bins = self.bins_of_objects(dirs, body[:-1])
        dist = 1 / np.linalg.norm(body[-1] - body[:-1], axis=1)
        ret = np.zeros(3, dtype=np.float)
        for i in range(3):
            cands = dist[bins == i]
            if len(cands):
                ret[i] = np.sum(cands) / len(cands)

        return ret

    def threats(self, safety_values):

        posses = self.surroundings_pixels(mode=1)
        for i in range(len(posses)):
            if safety_values[i]:
                safety_values[i] = int(posses[i] == self.body[0] or self.body_set[posses[i]] == 0)

        return safety_values

    def create_input(self):

        walls_dis = self.dis_to_walls()
        food_deg = np.abs(self.deg_to_food())
        food_dis = NN.put_in_range(ACTIVATIONS[SIGMOID](self.dis_to_food() * 4 / CROSS_VALUE))
        body_dis = NN.put_in_range(ACTIVATIONS[SIGMOID](self.dis_to_body() * 4))
        # food_deg = self.deg_to_food()
        # food_dis = NN.put_in_range(ACTIVATIONS[SIGMOID_NEW](self.dis_to_food()), 0, 1)
        # body_dis = ACTIVATIONS[SIGMOID](self.dis_to_body())
        danger = self.threats(walls_dis - 1)
        walls_dis = walls_dis / CROSS_VALUE
        if save_cand >= 3:
            print("walls", walls_dis, "food dis", food_dis, "food deg", food_deg, "body dis", body_dis, "danger", danger)
        ret = np.concatenate((walls_dis, body_dis, food_dis, food_deg, danger))
        return ret

    def forward(self):
        directions = self.available_steps()
        current_layer = self.create_input()[:self.ws[0].shape[0]]
        n_layers = len(self.ws)
        for i in range(n_layers):
            current_layer = current_layer.dot(self.ws[i]) + self.bs[i]
            if i < n_layers - 1: current_layer = (np.maximum(current_layer, 0))
        if save_cand >= 3:
            print(current_layer)
        max_val = np.max(current_layer)
        indices = current_layer == max_val
        optional_dir = directions[indices]
        return np.random.choice(optional_dir)

    def reward(self):
        s, a = self.total_steps, self.score
        # s /= 1 if a == 0 else a
        # self.rew = s + ((2 ** a) + 500 * (a ** 2.3 - self.death_penalty**1.1)) - (0.25 * (s ** 1.3) * (a ** 1.2))
        self.rew = s + ((2 ** a) + 500 * (a ** 2.3 - self.death_penalty**1.15)) - (0.25 * (s ** 1.4) * (a ** 1.2))
        # self.rew = s + ((2 ** a) + 500 * (a ** 2.3 - self.death_penalty**4)) - (0.50 * (s ** 1.3) * (a ** 1.2))

        return self.rew

    def update(self, d):

        xc, yc = d
        hx, hy = self.body[-1]
        new_head = (xc + hx, hy + yc)

        if self.has_eaten(self.food, new_head):
            self.add_cell(new_head)
            self.food = NN.generate_food(self)
            return True, False

        to_remove = self.body[0]
        self.body_set[to_remove] = 0
        self.body_set[new_head] += 1
        self.body.rotate(-1)
        self.body[-1] = new_head

        if self.bites_itself(new_head):
            self.death_penalty = 4
            return False, True

        elif self.hits_wall(*new_head):
            return False, True

        return False, False

    def run(self):
        self.score = 0
        self.total_steps = 0
        self.death_penalty = 0
        self.foods_so_far = []

        score = 0
        eaten, failed = False, False
        steps = 0
        self.set_body_and_food()
        # display = None
        # if WITH_GUI & (save_cand != 1):
        #     display = NN.init_display()
        #     NN.fill_display(display, self, self.food, score)
        f = self.food
        while True:

            if steps >= MAX_STEPS:
                self.death_penalty = 2
                break
            elif failed or score == MAXIMUM_SCORE:
                self.foods_so_far.append(tuple(f))
                break

            move = self.forward()
            self.curr_dir = move
            d = DIRECTION_TO_TUPLE[move]
            f = self.food
            eaten, failed = self.update(d)
            if eaten:
                self.foods_so_far.append(tuple(f))
                score += 1
                steps = -1
            steps += 1
            self.total_steps += 1

            # if WITH_GUI & (save_cand == 3):
            #     NN.fill_display(display, self, self.food, score)
        self.score = score
        # return score, self.total_steps, self.death_penalty
        return self


class NN(GameMode, SnakeGame):

    def __init__(self, n_threads=10):
        super().__init__()
        self.generation = [NNSnake() for _ in range(GEN_SIZE)]
        self.n_threads = n_threads
        self.best_candidate = self.generation[0]
        self.data = []
        self.mutation_rate = MUTATION_THRESH
        self.gen_size = GEN_SIZE

    @staticmethod
    def decoy():
        food = (2, 3)
        head = (2, 2)
        return food, head

    def mate(self):

        def roulette_wheel(curr_pop):
            rewards = [s.rew for s in curr_pop]
            wheel_sum = np.sum(rewards)
            if not wheel_sum:
                return deepcopy(curr_pop[np.random.choice(np.arange(len(curr_pop)))])

            pick = np.random.uniform(0, wheel_sum)
            idx = np.argmax(np.cumsum(rewards) >= pick)
            return deepcopy(curr_pop[idx])

        def get_offsprings(x: NNSnake, y: NNSnake, par_chance=0.6):

            for i in range(len(x.ws)):
                w_change, b_change = np.random.rand(*x.ws[i].shape) < par_chance, np.random.rand(*x.bs[i].shape) < par_chance

                x.ws[i][~w_change] = y.ws[i][~w_change]
                y.ws[i][w_change] = x.ws[i][w_change]
                x.bs[i][~b_change] = y.bs[i][~b_change]
                y.bs[i][b_change] = x.bs[i][b_change]

            return [x, y]

        offsp = []
        for i in range(len(self.generation)):
            pair = [deepcopy(self.generation[i]), roulette_wheel(self.generation)]
            offsp += get_offsprings(*sorted(pair, key=lambda s: s.rew))
        np.random.shuffle(offsp)
        self.generation += offsp

    def mutate(self):
        for s in self.generation[self.gen_size:]:
            for i in range(len(s.ws)):

                h, w = s.ws[i].shape
                indices = np.random.rand(h, w) < self.mutation_rate
                gauss_values = np.random.normal(size=(h, w))
                s.ws[i][indices] += gauss_values[indices]

                h = len(s.bs[i])
                indices = np.random.rand(h) < self.mutation_rate
                gauss_values = np.random.normal(size=h)
                s.bs[i][indices] += gauss_values[indices]

    @staticmethod
    def function_f(s):
        return s.run()

    def evaluate(self, pool=None):
        # self.generation += [deepcopy(self.best_candidate) for _ in range(12)]
        [s.set_body_and_food() for s in self.generation]
        self.generation = pool.map(NN.function_f, self.generation) if POOL else [s.run() for s in self.generation]
        self.generation.sort(reverse=True, key=lambda a: a.reward())
        self.generation = self.generation[:self.gen_size]

    def train(self, number_of_rounds=60):

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.best_candidate.rew = 0
        best_score, turns = 0, 0
        for j in range(number_of_rounds):
            start = time.time()
            self.mate()
            self.mutate()
            self.evaluate(pool)
            self.data.append(self.generation[0].score)
            if self.generation[0].score >= self.best_candidate.score:
                if self.generation[0].score > self.best_candidate.score: turns = 0
                self.best_candidate = deepcopy(self.generation[0])
                best_score = self.best_candidate.score
                self.gen_size = GEN_SIZE
                self.mutation_rate = MUTATION_THRESH
                with open("best.pickle", "wb") as file:
                    pickle.dump(self.best_candidate, file)
            turns += 1

            if turns % 10 == 0:
                new = [NNSnake() for _ in range(20)]
                for s in new:
                    s.sore = best_score

                self.generation = self.generation + new
                self.gen_size += 5
                self.mutation_rate += 0.1

            print(f"Iter {j}: ", "{:.2f}".format(self.generation[0].score), best_score, turns, time.time() - start)

        with open("best.pickle", "wb") as file:
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
                temp = deepcopy(curr_snake)
                # curr_snake.create_input()
                move = curr_snake.forward()
                curr_snake.curr_dir = move
                d = DIRECTION_TO_TUPLE[move]
                eat, fails = curr_snake.update(d)
                if WITH_GUI:
                    self.fill_display(display, curr_snake, curr_snake.food, score)

                if eat:
                    score += 1
                    steps = 0
                else:
                    steps += 1
                self.pygameMUST()
                self.clock.tick(50)

                if fails or steps > MAX_STEPS:
                    if WITH_GUI:
                        temp.create_input()
                        self.exit(display, score)
                        self.clock.tick(0.3)
                    break

    def replay_best(self, number_of_games=1):

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
                if eat:
                    score += 1
                    steps = 0
                    curr_snake.food = curr_snake.foods_so_far[score]
                else:
                    steps += 1

                if WITH_GUI:
                    self.fill_display(display, curr_snake, curr_snake.food, score)
                self.pygameMUST()
                self.clock.tick(50)

                if fails or steps > MAX_STEPS:
                    if WITH_GUI:
                        self.exit(display, score)
                        self.clock.tick(0.3)
                    break

    def decay_value(self, value, factor=0.997):
        return value * factor

    @staticmethod
    def get_truncated_normal(mean=0.1, sd=0.9, size=10):
        low = max(0.04, mean - sd)
        upp = mean + sd
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(size=size)

    @staticmethod
    def activation(x, activation=RELU):
        assert activation in ACTIVATIONS
        return ACTIVATIONS[activation](x)

    @staticmethod
    def put_in_range(arr, newMin=0, newMax=1):
        arr = np.array(arr)
        oldMax, oldMin = np.max(arr), np.min(arr)
        oldRange = oldMax - oldMin
        NewRange = newMax - newMin
        return (arr - oldMin) * NewRange / oldRange + newMin if oldRange else arr


def save_files(im="res"):
    from scipy.signal import savgol_filter

    data = np.array(nn.data)
    titles = ["best_cand_score"]

    # yhat = savgol_filter(data[:, i], len(data[:, i]), 3)  # window size 51, polynomial order 3
    plt.plot(np.arange(len(data)), data, label="best_cand_score")
    # plt.plot(np.arange(len(data[:, i])), yhat, label=titles[i] + " Avg")


    plt.legend()
    name = f"NNsnakeResults\\res{f_counter}.jpg"
    plt.savefig(name)

    with open(f"NNsnakeResults\\best{f_counter}.pickle", "wb") as file:
        pickle.dump(nn.best_candidate, file)

    with open(f"NNsnakeResults\\data{f_counter}.pickle", "wb") as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    x = 100
    y = 100
    import os

    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)

    # import pygame

    pygame.init()
    screen = pygame.display.set_mode((100, 100))

    nn = NN(n_threads=N_THREADS)

    if save_cand == 1:
        nn.train(number_of_rounds=NUM_TRAINING_ROUNDS)
        save_files()

    elif save_cand == 2:
        with open("best.pickle", "rb") as file:
            nn.best_candidate = pickle.load(file)
            # nn.play_with_winner(number_of_games=100)
            nn.generation = [deepcopy(nn.best_candidate) for _ in range(GEN_SIZE)]
            nn.train(number_of_rounds=NUM_TRAINING_ROUNDS)
            save_files()
    elif save_cand == 3:

        with open(f"best.pickle", "rb") as file:
            nn.best_candidate = pickle.load(file)
            nn.play_with_winner(number_of_games=100)
    else:
        with open(f"best.pickle", "rb") as file:
            nn.best_candidate = pickle.load(file)
            nn.replay_best()


    pygame.quit()













