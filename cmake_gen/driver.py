import os
import json
import time
import logging
import pathlib
import random
import numpy as np
import heapq
import xgboost as xgb
from typing import List, Set

COMMAND_PREFIX = "srun -p V100 --gres=gpu:v132p:1 --exclusive "
# COMMAND_PREFIX = ""
DATA_PATH = str(pathlib.Path(__file__).parent.absolute() / ".." / "dataset")
BUILD_PATH = str(pathlib.Path(__file__).parent.absolute() / ".." / "build")
PARAM_PATH = "param.json"
PARAM_VAL = {
    "USE_ARRAY": [0, 1],
    "THREADS_PER_BLOCK": [32, 64, 128, 256, 512, 1024],
    "NUM_BLOCKS": [32, 64, 128, 256, 512, 1024],
    "IEP_BY_SM": [0, 1]}


logger = logging.getLogger("tuner")

def config_random_walk(config: dict) -> dict:
    '''
    Randomly walks to another configuration.
    '''
    key_list = list(PARAM_VAL.keys())
    from_i = random.choice(key_list)
    to_v = random.choice(PARAM_VAL[from_i])
    config[from_i] = to_v
    return config


def dict2list(params: dict) -> list:
    '''
    Convert `params` into a list
    '''
    return [val for key, val in params.items()]


class Manipulator():
    def __init__(self):
        self.params = self.read_params()
        self.xs = []
        self.ys = []
        self.best_config = ({}, float('inf'))
        self.trials = []
        self.bst = None
        self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "reg:linear",
            }
        self.batch_size = 10

        # global xgb
        # if xgb is None:
        #     xgb = __import__("xgboost")


    def read_params(self) -> dict:
        # read parameters
        with open(PARAM_PATH) as f:
            param_dict = json.load(f)
            assert type(param_dict) == type(dict()), "Param JSON file error!"
        
        return param_dict


    def random_configuration(self) -> dict:
        '''
        return a random configuration to be tested
        '''
        ret = {}
        while True:
            for key, item in PARAM_VAL.items():
                ret[key] = random.choice(item)
            if True:
                break
        return ret


    def next_batch(self, batch_size) -> List[dict]:
        '''
        Return a batch of configurations to be tested.
        '''
        ret = []
        trial_idx = 0
        while len(ret) < batch_size:
            while trial_idx < len(self.trials):
                trial = self.trials[trial_idx]
                if trial not in self.xs:
                    break
                trial_idx += 1
            else:
                trial_idx = -1
            
            chosen_config = self.trials[trial_idx] if trial_idx != -1 else self.random_configuration()
            ret.append(chosen_config)

        return ret

    
    def update(self, inputs: List[dict], results: List[float]) -> None:
        '''
        Add a test result to the manipulator.
        '''
        for params, res in zip(inputs, results):
            if res < self.best_config[1]:
                self.best_config = (params, res)

            self.xs.append(params)
            self.ys.append(res)
            # TODO: add these results to xgboost

        if self.bst is None:
            dtrain = xgb.DMatrix(np.asanyarray([dict2list(item) for item in self.xs]), np.asanyarray(self.ys))
            self.bst = xgb.train(self.xgb_params, dtrain)
        else:
            self.fit(inputs, results)
        

        self.trials = self.find_maximums(len(inputs), 100, 2)
        

    def fit(self, data_x: list, data_y: list):
        tic = time.time()
        index = np.random.permutation(len(data_x))
        dx = [dict2list(data) for data in data_x]
        dy = data_y

        dtrain = xgb.DMatrix(np.asanyarray(dx)[index], np.array(dy)[index])

        self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=2)
        
        logger.log(logging.INFO,
            "XGB train: %.2f\tobs: %d",
            time.time() - tic,
            len(data_x),
        )


    def predict(self, data_x: List[dict]):
        dtest = xgb.DMatrix(np.asanyarray([dict2list(item) for item in data_x]))
        return self.bst.predict(dtest)


    def find_maximums(self, num: int, n_iter: int, log_interval: int):
        '''
        Find the best `num` sets of parameters
        '''
        class Pair():
            '''
            class for heapifying tuple[float, dict]
            '''
            def __init__(self, a, b) -> None:
                self.first = a
                self.second = b
            
            def __lt__(self, other) -> bool:
                return self.first < other.first
            
            def __gt__(self, other) -> bool:
                return self.first > other.first

        tic = time.time()
        temp = 100

        points = [self.random_configuration() for _ in range(num)]

        scores = self.predict(points)

        heap_items = [Pair(float("-inf"), self.random_configuration()) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = [x.second for x in heap_items]

        for _ in range(n_iter):
            new_points = np.empty_like(points)
            for i in range(len(new_points)):
                new_points[i] = config_random_walk(points[i])
            new_scores = self.predict(new_points)

            ac_prob = np.exp(np.minimum((new_scores - scores) / (temp + 1e-5), 1))  # accept probability
            ac_index = np.random.random(len(ac_prob)) < ac_prob                     # accepted index

            for idx in range(len(ac_prob)):         # update accepted points and scores
                if ac_index[idx]:
                    points[idx] = new_points[idx]
                    scores[idx] = new_scores[idx]

            for score, point in zip(new_scores, new_points):
                if score > heap_items[0].first and point not in in_heap:
                    pop = heapq.heapreplace(heap_items, Pair(score, point))
                    in_heap.remove(pop.second)
                    in_heap.append(point)

            temp *= 0.9

            if log_interval and _ % log_interval == 0:
                logger.log(logging.INFO,
                    f"\rFinding maximums... {(_ / n_iter):.2f}%, time elapsed: {(time.time() - tic):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}"
                )

        logger.log(logging.INFO,
            f"\rFinding maximums... 100%, time elapsed: {(time.time() - tic):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}"    
        )

        return [x.second for x in heap_items]


class Tuner:
    def __init__(self, job:str, options:str, manipulator: Manipulator) -> None:
        '''
        :param: job cuda program
        :param: options data path and pattern size
        '''
        self.job = job
        self.options = options
        self.manipulator = manipulator


    def tune(self, max_round: int, k: int):
        '''
        The final autotune interface
        '''
        logger.log(logging.INFO, "Start tuning.")

        if len(self.manipulator.trials) == 0:
            configs = [self.manipulator.random_configuration() for _ in range(k)]
        else:
            configs = self.manipulator.trials

        for _ in range(max_round):
            res = [self.compile_and_run(config) for config in configs]
            print(res)
            self.manipulator.update(configs, res)
            configs = self.manipulator.trials
            logger.log(logging.INFO, f"Round {_} / {max_round} Best performance: {self.manipulator.best_config[1]}s")

        return self.manipulator.best_config


    def compile_and_run(self, param_dict: dict) -> float:
        '''
        Compile using the given patameters and return the time cost.
        '''
        command = ""

        for key, val in param_dict.items():
            command += f"-D{key}={val} "

        print(BUILD_PATH)
        os.chdir(BUILD_PATH)
        ret_code = os.system("cmake --quiet " + command + ".. >" + os.devnull)
        assert ret_code == 0, f"CMake exited with non-zero code {ret_code}"
        ret_code = os.system("make -j -s > /dev/null")
        assert ret_code == 0, f"Make exited with non-zero code {ret_code}"

        # run
        os.chdir(BUILD_PATH + "/bin")

        if manip.best_config[1] != float('inf'):
            t0 = time.time()
            ret_code = os.system(f"timeout {manip.best_config[1] * 2} " + COMMAND_PREFIX + "./" + " ".join([self.job, self.options]) + " > /dev/null")
            t1 = time.time()
        else:
            t0 = time.time()
            ret_code = os.system(COMMAND_PREFIX + "./" + " ".join([self.job, self.options]) + " > /dev/null")
            t1 = time.time()


        if ret_code != 0:
            print(f"Graph mining program returned non-zero code {ret_code}")
            return float('inf')

        return (t1 - t0)


if __name__ == "__main__":
    manip = Manipulator()
    tuner = Tuner("gpu_mc", f"{DATA_PATH}/wiki-vote.g 3", manip)
    tuner.tune(5, 2)
    print(manip.best_config[0])
    print(f"Time cost: {manip.best_config[1]}")
    # print(f"Time cost: {time_cost}")
