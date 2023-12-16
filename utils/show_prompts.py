"""
Usage:
    python show_plan_generations.py generations.pkl
"""

import pickle
import sys
if len(sys.argv) == 2:
    out_file = sys.argv[1]
else:
   print("Provide the .pkl file containing the generations as the only arg.")
   exit()
with open(out_file, "rb") as fp:   # Unpickling
  prompts = pickle.load(fp)

for i, prompt in enumerate(prompts):
    print(f'************************{i}**************************************\n'
          f'{prompt}')