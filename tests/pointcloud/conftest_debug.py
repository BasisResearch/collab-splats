
import sys, pathlib
proj = str(pathlib.Path(__file__).parents[2])
print("DEBUG proj:", proj)
print("DEBUG in path:", proj in sys.path)
print("DEBUG sys.path[:3]:", sys.path[:3])
import importlib.util
spec = importlib.util.find_spec('evals')
print("DEBUG evals spec:", spec)
