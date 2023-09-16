import os
import json

COMMAND_PREFIX = "nvcc --gpu-architecture=sm_50 test.cu -o test"

with open("param.json") as f:
    param_dict = json.load(f)
    assert type(param_dict) == type(dict()), "Param JSON file error!"

command = ""

# read params
for key, val in param_dict.items():
    command += f"-D{key}={val} "

os.chdir("../build")
os.system("cmake " + command + "..")
os.system("make -j")
