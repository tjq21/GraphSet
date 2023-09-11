import os
import json

COMMAND_PREFIX = "nvcc --gpu-architecture=sm_50 test.cu -o test"

with open("param.json") as f:
    param_dict = json.load(f)
    assert type(param_dict) == type(dict()), "Param JSON file error!"

definitions = ""

# read set-storage
try:
    set_storage = param_dict["set-storage"]
    definitions += f"-D{set_storage.upper()} "
except Exception as e:
    definitions += "-DARRAY "

with open("../CMakeLists.txt", "r") as f:
    content = f.readlines()
    f.close()
content.append(f"ADD_DEFINITIONS({definitions})")
with open("../CMakeLists.txt", "w") as f:
    f.writelines(content)
