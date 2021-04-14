import multiprocessing
import pickle
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
        self.ws = [np.random.normal(size=(INPUT_SIZE, HIDDEN_SIZE_1)),
                   np.random.normal(size=(HIDDEN_SIZE_1, HIDDEN_SIZE_2)),
                   np.random.normal(size=(HIDDEN_SIZE_2, OUTPUT_SIZE))]
        self.bs = [np.zeros(HIDDEN_SIZE_1),
                   np.zeros(HIDDEN_SIZE_2),
                   np.zeros(OUTPUT_SIZE)]
        self.score = 0
        self.total_steps = 0
        self.death_penalty = 0
        self.rew = 0
        self.round = 0

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
        boundaries = np.array([[0, 0], [0, 0], [GAME_WIDTH, 0], [0, GAME_HEIGHT]])
        availabe_s = self.available_steps()
        head = np.array(self.body[-1])
        moves = DIRECTION_TO_TUPLE[availabe_s]
        dis_to_walls = np.linalg.norm(boundaries[availabe_s] - np.multiply(moves, head), axis=1)
        return dis_to_walls

    def dis_to_food(self):
        posses = np.array([np.array(self.body[-1]) + np.array(DIRECTION_TO_TUPLE[d]) for d in self.available_steps()])
        ret = np.linalg.norm(posses - np.array(self.food), axis=1)
        return ret

    def deg_to_food(self):
        food_deg = self.deg_to_objects(DIRECTION_TO_ANGLE[self.available_steps()], self.food)
        food_deg = np.abs(food_deg)
        return food_deg

    def dis_to_body(self):
        if len(self.body) < 3:
            return np.zeros(3)

        body = np.array(self.body)
        dirs = self.available_steps()
        bins = self.bins_of_objects(dirs, body[:-1])
        ret = np.zeros(3, dtype=np.float)
        for i in range(3):
            cands = body[:-1][bins == i]
            if len(cands):
                ret[i] = len(cands) * np.linalg.norm(body[-1] - cands) / (len(bins))
        return NN.put_in_range(ret, 0, 1)

    def threats(self):
        return [self.dis_from_threat(d, self.food) for d in self.available_steps()]

    def create_input(self):

        walls_dis = self.dis_to_walls() / CROSS_VALUE
        food_dis = self.dis_to_food() / CROSS_VALUE
        food_deg = self.deg_to_food()  #/ PI
        body_dis = self.dis_to_body()
        danger = self.threats()

        if save_cand == 3:
            print("walls", walls_dis, "food dis", food_dis, "food deg", food_deg, "body dis", body_dis, "danger", danger)
        ret = np.concatenate((walls_dis, body_dis, food_dis, food_deg, danger))
        return ret

    def forward(self):
        directions = self.available_steps()
        current_layer = self.create_input()[:self.ws[0].shape[0]]
        n_layers = len(self.ws)
        for i in range(n_layers):
            current_layer = (current_layer.dot(self.ws[i]) + self.bs[i])
            if i < n_layers - 1: current_layer = np.maximum(current_layer, 0)
        if save_cand == 3:
            print(current_layer)
        max_val = np.max(current_layer)
        indices = current_layer == max_val
        optional_dir = directions[indices]
        return np.random.choice(optional_dir)

    def reward(self):
        s, a = self.total_steps, self.score
        self.rew = s + ((2 ** a) + 500 * (a ** 2.3)) - (0.25 * (s ** 1.3) * (a ** 1.2))
        # self.rew = a + s - self.death_penalty
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
            self.death_penalty = -4
            return False, True

        elif self.hits_wall(*new_head):
            return False, True

        return False, False

    def run(self):
        self.score = 0
        self.total_steps = 0
        self.death_penalty = 0

        score = 0
        eaten, failed = False, False
        steps = 0
        self.set_body_and_food()
        # display = None
        # if WITH_GUI & (save_cand != 1):
        #     display = NN.init_display()
        #     NN.fill_display(display, self, self.food, score)

        while True:

            if steps >= MAX_STEPS:
                self.death_penalty = -2
                break
            elif failed:
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

            # if WITH_GUI & (save_cand == 3):
            #     NN.fill_display(display, self, self.food, score)
        self.score = score
        return self

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

        def get_offsprings(x: NNSnake, y: NNSnake, par_chance=0.7):

            for i in range(len(x.ws)):
                w_change, b_change = np.random.rand(*x.ws[i].shape) < par_chance, np.random.rand(*x.bs[i].shape) < par_chance

                x.ws[i][~w_change] = y.ws[i][~w_change]
                y.ws[i][w_change] = x.ws[i][w_change]
                x.bs[i][~b_change] = y.bs[i][~b_change]
                y.bs[i][b_change] = x.bs[i][b_change]

            return [x, y]

        offsp = []
        for i in range(GEN_SIZE):
            pair = [deepcopy(self.generation[i]), roulette_wheel(self.generation)]
            offsp += get_offsprings(*sorted(pair, key=lambda s: s.rew))
        np.random.shuffle(offsp)
        self.generation += offsp

    def mutate(self):
        for s in self.generation[GEN_SIZE:]:
            for i in range(len(s.ws)):

                h, w = s.ws[i].shape
                indices = np.random.rand(h, w) < MUTATION_THRESH
                gauss_values = np.random.normal(size=(h, w))
                s.ws[i][indices] = s.ws[i][indices] + gauss_values[indices]

                h = len(s.bs[i])
                indices = np.random.rand(h) < MUTATION_THRESH
                gauss_values = np.random.normal(size=h)
                s.bs[i][indices] = s.bs[i][indices] + gauss_values[indices]

    @staticmethod
    def function_f(s):
        return s.run()

    def evaluate(self, pool=None):
        [s.set_body_and_food() for s in self.generation]
        if POOL:
            self.generation = pool.map(NN.function_f, self.generation)
        else:
            self.generation = [s.run() for s in self.generation]

        self.generation.sort(reverse=True, key=lambda a: a.reward())
        self.generation = self.generation[:GEN_SIZE]
        curr_best_snake_avg = self.generation[0].score
        print("{:.2f}".format(curr_best_snake_avg), end="")
        self.data.append(curr_best_snake_avg)

    def train(self, number_of_rounds=60):

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.best_candidate.rew = 0
        best_score, turns = 0, 0
        for j in range(number_of_rounds):
            # np.random.seed()
            self.mate()
            self.mutate()
            print(f"Iter {j}: ", end="")
            self.evaluate(pool)

            if self.generation[0].score > self.best_candidate.score:
                self.best_candidate = deepcopy(self.generation[0])
                best_score = self.best_candidate.score
                turns = 0

            if int(j * 0.1) == j * 0.1:
                with open("best.pickle", "wb") as file:
                    pickle.dump(self.best_candidate, file)
            turns += 1
            print(" ", best_score, turns)

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
                        self.exit(display, score)
                        self.clock.tick(0.3)
                    break

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
    def put_in_range(arr, newMin, newMax):
        arr = np.array(arr)
        oldMax, oldMin = np.max(arr), np.min(arr)
        oldRange = oldMax - oldMin
        NewRange = newMax - newMin
        return (arr - oldMin) * NewRange / oldRange + newMin if oldRange else arr


def save_files(im="res"):
    from scipy.signal import savgol_filter

    f_counter = 0

    files = glob.glob("./**/*.jpg")
    if len(files):
        for f in files:
            x = max([int(n) for n in re.findall(r"\d+", f)])
            if x > f_counter:
                f_counter = x

    f_counter += 1
    data = np.array(nn.data)
    titles = ["best_cand_score", "gen_avg"]
    for i in range(data.shape[1]):
        # yhat = savgol_filter(data[:, i], len(data[:, i]), 3)  # window size 51, polynomial order 3
        plt.plot(np.arange(len(data[:, i])), data[:, i], label=titles[i])
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
        with open("NNsnakeResults/best3.pickle", "rb") as file:
            nn.best_candidate = pickle.load(file)
            # nn.play_with_winner(number_of_games=100)
            nn.generation = [deepcopy(nn.best_candidate) for _ in range(GEN_SIZE)]
            nn.train(number_of_rounds=NUM_TRAINING_ROUNDS)
            save_files()
    else:

        with open(f"best.pickle", "rb") as file:
            nn.best_candidate = pickle.load(file)
            nn.play_with_winner(number_of_games=100)

    pygame.quit()













