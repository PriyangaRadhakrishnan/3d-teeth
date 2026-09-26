import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.obj_loader import inspect_obj


parser = argparse.ArgumentParser()
parser.add_argument("obj")
parser.add_argument("--json")
args = parser.parse_args()
result = inspect_obj(args.obj, args.json)
print(json.dumps(result, indent=2))