"""
Usage:
    python show_generations.py generations.pkl
"""

import pickle
import sys
if len(sys.argv) == 2:
    out_file = sys.argv[1]
else:
   print("Provide the .pkl file containing the generations as the only arg.")
   exit()
with open(out_file, "rb") as fp:   # Unpickling
  mylist_gens = pickle.load(fp)

for i, task in enumerate(mylist_gens):
  for j, attempt in enumerate(task):
    print(f'==============================\n{i}.{j}\n{attempt}')