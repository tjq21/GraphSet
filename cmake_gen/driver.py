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
import argparse

FLOAT_INF = 1e38

COMMAND_PREFIX = "srun -p V100 --gres=gpu:v132p:1 --exclusive "
# COMMAND_PREFIX = ""
DATA_PATH = pathlib.Path(__file__).parent.absolute() / ".." / "dataset"
BUILD_PATH = pathlib.Path(__file__).parent.absolute() / ".." / "build"
PARAM_PATH = pathlib.Path(__file__).parent.absolute() / "param.json"
CONF_PATH = pathlib.Path(__file__).parent.absolute() / "best_config.json"
RESULT_PATH = pathlib.Path(__file__).parent.absolute() / "counting_time_cost.txt"
RECORD_PATH = pathlib.Path(__file__).parent.absolute() / "record.json"
PARAM_VAL = {
    "USE_ARRAY": [1],
    "THREADS_PER_BLOCK": [32, 64, 128, 256, 512, 1024],
    "NUM_BLOCKS": [32, 64, 128, 256, 512, 1024],
    "IEP_BY_SM": [0, 1],
    "MAXREG": [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "SORT": [0, 1],
    "PARALLEL_INTERSECTION": [0, 1],
    "BINARY_SEARCH": [0, 1]
}

JOB_NAME = "gpu_graph"

logger = logging.getLogger("tuner")
parser = argparse.ArgumentParser()
parser.add_argument('-r', type=bool)
parser.add_argument('data', type=str)
parser.add_argument('graph_size', type=str)
parser.add_argument('pattern_string', type=str)
parser.add_argument('use_iep', type=str, help='<0/1> Use IEP or not')
parser.add_argument('--debug_msg', default=False, action='store_true', help='Enable output of CMake and Make')
parser.add_argument('--run_best_config', default=False, action='store_true', help='Run the best configuration only')
parser.add_argument('--run_default_config', default=False, action='store_true', help='Run the default configuration only')

def config_random_walk(config: dict) -> dict:
    '''
    Randomly walks to another configuration.
    '''
    ret = {}
    for key, val in config.items():
        ret[key] = val
    key_list = list(PARAM_VAL.keys())
    from_i = random.choice(key_list)
    to_v = random.choice(PARAM_VAL[from_i])
    while to_v == config[from_i]:
        from_i = random.choice(key_list)
        to_v = random.choice(PARAM_VAL[from_i])
    ret[from_i] = to_v
    return ret


def dict2list(params: dict) -> list:
    '''
    Convert `params` into a list
    '''
    return [val for key, val in params.items()]


class Manipulator():
    def __init__(self, num_warmup_sample: int = 100):
        # self.params = self.read_params(PARAM_PATH)
        self.xs = []
        self.ys = []
        self.best_config = ({}, FLOAT_INF)
        self.trials = []
        self.bst = None
        self.num_warmup_sample = num_warmup_sample
        self.xgb_params = {
            "max_depth": 10,
            "gamma": 0.001,
            "min_child_weight": 0,
            "eta": 0.2,
            "verbosity": 0,
            "disable_default_eval_metric": 1,
        }
        self.batch_size = 10

        # global xgb
        # if xgb is None:
        #     xgb = __import__("xgboost")

        with open(RECORD_PATH, "r") as f:
            if os.path.getsize(RECORD_PATH) != 0:
                records = json.load(f)
                for res in records:
                    self.xs.append(res[0])
                    self.ys.append(res[1])
        assert len(self.xs) == len(self.ys)
        print(f"Loaded {len(self.ys)} records from file.")


    def read_params(self, param_path: str) -> dict:
        # read parameters
        with open(param_path) as f:
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
            if ret not in self.xs:
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

    
    def update(self, k:int, inputs: List[dict], results: List[float]) -> None:
        '''
        Add a test result to the manipulator.
        XGBoost does not support additional traning, so re-train a model each time.
        '''
        print("input:", inputs, results)
        if len(inputs) == 0:
            return

        for params, res in zip(inputs, results):
            if res < self.best_config[1]:
                self.best_config = (params, res)

            self.xs.append(params)
            self.ys.append(res)

        with open(RECORD_PATH, "w") as f:
            json.dump([[x, y] for x, y in zip(self.xs, self.ys)], f)

        if self.bst is None:
            tmp_matrix = np.asanyarray([dict2list(item) for item in self.xs])
            print("tmp_matrix:", tmp_matrix)
            perm = np.random.permutation(len(self.xs))
            dtrain = xgb.DMatrix(tmp_matrix[perm], np.asanyarray(self.ys)[perm])
            self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=1000)
        else:
            self.fit(self.xs, self.ys)

        possible_vals = []
        print("\nAfter update:")
        for ua in PARAM_VAL['USE_ARRAY']:
            for tpb in PARAM_VAL['THREADS_PER_BLOCK']:
                for nb in PARAM_VAL['NUM_BLOCKS']:
                    for ibs in PARAM_VAL["IEP_BY_SM"]:
                        for maxreg in PARAM_VAL["MAXREG"]:
                            tmp = [ua, tpb, nb, ibs, maxreg, 0]
                            tmp_matrix = xgb.DMatrix(np.asanyarray([tmp]))
                            tmp_val = self.bst.predict(tmp_matrix)[0]
                            if tmp_val not in possible_vals:
                                possible_vals.append(tmp_val)
        print("Possible values:", possible_vals)

        self.trials = self.find_maximums(k, 40, 5)
        

    def fit(self, data_x: list, data_y: list):
        if len(data_x) == 0:
            return

        tic = time.time()
        index = np.random.permutation(len(data_x))
        dx = [dict2list(data) for data in data_x]
        dy = data_y

        dtrain = xgb.DMatrix(np.asanyarray(dx)[index], np.array(dy)[index])
        self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=10000)

        print(
            "XGB train: %.2f\tobs: %d",
            time.time() - tic,
            len(data_x),
        )


    def predict(self, data_x: List[dict]):
        if len(self.xs) < self.num_warmup_sample:
            return np.random.uniform(0, 1, len(data_x))     # TODO: add a better range of random score
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
            
            # reversed comparison to make max heap
            def __lt__(self, other) -> bool:
                return self.first > other.first
            
            def __gt__(self, other) -> bool:
                return self.first < other.first

        tic = time.time()
        temp = 0.1

        points = [self.random_configuration() for _ in range(num)]

        scores = self.predict(points)

        heap_items = [Pair(scores[i], points[i]) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = [x.second for x in heap_items]

        for _ in range(n_iter):
            new_points = np.empty_like(points)
            for i in range(len(new_points)):
                new_points[i] = config_random_walk(points[i])
            new_scores = self.predict(new_points)

            ac_prob = np.exp((scores - new_scores) / temp)                          # accept probability
            ac_index = np.random.random(len(ac_prob)) < ac_prob                     # accepted index

            for idx in range(len(ac_prob)):         # update accepted points and scores
                if ac_index[idx]:
                    points[idx] = new_points[idx]
                    scores[idx] = new_scores[idx]

            for score, point in zip(new_scores, new_points):
                if score < heap_items[0].first and point not in in_heap:
                    pop = heapq.heapreplace(heap_items, Pair(score, point))
                    in_heap.remove(pop.second)
                    in_heap.append(point)

            temp *= 0.9

            if log_interval and _ % log_interval == 0:
                print(f"\rFinding maximums... {(_ / n_iter):.2f}%, time elapsed: {(time.time() - tic):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}")
                logger.log(logging.INFO,
                    f"\rFinding maximums... {(_ / n_iter):.2f}%, time elapsed: {(time.time() - tic):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}"
                )

        print(f"\rFinding maximums... {(_ / n_iter):.2f}%, time elapsed: {(time.time() - tic):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}")
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
        self.best_time = FLOAT_INF


    def tune(self, max_round: int, k: int, debug_msg: bool):
        '''
        The final autotune interface
        '''
        logger.log(logging.INFO, "Start tuning.")

        if len(self.manipulator.trials) == 0:
            configs = [self.manipulator.random_configuration() for _ in range(k)]  # 10 samples for warm-up trial
        else:
            configs = self.manipulator.trials

        for _ in range(max_round):
            print(f"Tuning round {_+1}/{max_round}")
            valid_configs = []
            results = []
            for config in configs:
                time_cost = self.compile_and_run(config, debug_msg)
                if time_cost != FLOAT_INF:
                    valid_configs.append(config)
                    results.append(time_cost)
            if len(valid_configs) == 0:
                print("Too many resources required. Stopping...")
                break
            self.manipulator.update(k, valid_configs, results)
            configs = self.manipulator.trials
            print("len(valid_configs) =",len(valid_configs))
            print(f"Round {_+1} / {max_round} Best performance: {self.manipulator.best_config[1]:.2f}s")
            logger.log(logging.INFO, f"Round {_+1} / {max_round} Best performance: {self.manipulator.best_config[1]:.2f}s")

        return self.manipulator.best_config


    def compile_and_run(self, param_dict: dict, debug_msg: bool) -> float:
        '''
        Compile using the given patameters and return the time cost.
        '''
        print(f"config = {param_dict}")

        definitions = ""

        for key, val in param_dict.items():
            definitions += f"-D{key}={val} "

        print(BUILD_PATH)
        os.chdir(BUILD_PATH)
        if debug_msg:
            command = "cmake " + definitions + ".."
        else:
            print("Generating makefile with CMake...")
            command = "cmake " + definitions + ".. > /dev/null 2>&1"
        ret_code = os.system(command)
        assert ret_code == 0, f"CMake exited with non-zero code {ret_code}"

        if debug_msg:
            command = "make -j"
        else:
            print("Compiling with Make...")
            command = "make -j > /dev/null 2>&1"
        ret_code = os.system(command)
        assert ret_code == 0, f"Make exited with non-zero code {ret_code}"

        # run
        os.chdir(BUILD_PATH / "bin")

        print("Running...")

        # if self.best_time != FLOAT_INF:
        #     ret_code = os.system(f"timeout {self.best_time * 2} " + COMMAND_PREFIX + "./" + " ".join([self.job, self.options]) + " > /dev/null")
        # else:
        ret_code = os.system(COMMAND_PREFIX + "./" + " ".join([self.job, self.options]) + " > /dev/null")

        if ret_code != 0:
            print(f"Graph mining program returned non-zero code {ret_code}")
            return FLOAT_INF

        with open(RESULT_PATH, "r") as f:
            time_cost = float(f.readline())

        if time_cost < self.best_time:
            self.best_time = time_cost

        print(f"Time cost: {time_cost:.2f}s\n")

        return time_cost


if __name__ == "__main__":

    args = parser.parse_args()

    '''
    pattern string: 
        if(buffer[INDEX(i,j,size)] == '1')
            add_edge(i,j);
    '''

    for file in [CONF_PATH, PARAM_PATH, RESULT_PATH, RECORD_PATH]:
        if not file.is_file():
            with open(file, "w"):
                pass

    manip = Manipulator()
    tuner = Tuner(JOB_NAME, f"{DATA_PATH}/{args.data} " + " ".join([args.graph_size, args.pattern_string, args.use_iep]), manip)
    if args.run_best_config:
        param_dict = manip.read_params(CONF_PATH)
        tuner.compile_and_run(param_dict, True)
    elif args.run_default_config:
        param_dict = {
            "USE_ARRAY": 1,
            "THREADS_PER_BLOCK": 1024,
            "NUM_BLOCKS": 128,
            "IEP_BY_SM": 1,
            "LIMIT_REG": 1,
            "MAXREG": 64,
            "SORT": 0,
            "PARALLEL_INTERSECTION": 1,
            "BINARY_SEARCH": 1}
        tuner.compile_and_run(param_dict, True)
    else:
        best_config = tuner.tune(10, 3, debug_msg=args.debug_msg)
        # best_config = tuner.manipulator.find_maximums(5, 50, 1)
        print("Best configuration:", best_config[0])
        print(f"Estimated time cost: {best_config[1]:.2f}s")
        with open(CONF_PATH, "w") as f:
            json.dump(best_config[0], f, indent=4)
        print("Best configuration dumped in ./best_config.json")
