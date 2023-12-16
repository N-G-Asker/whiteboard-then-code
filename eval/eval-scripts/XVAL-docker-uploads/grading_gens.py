"""
This script is adapted from part of example_script.py on HuggingFace's public repository for the APPS  Metric: 
https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/example_script.py

Specifically, this code reuses the boiler-plate code for loading HF datasets and evaluating, by accessing (publically-available, open-source) repos on the HF Hub dedicated to a benchmark/metric of interest.

For the purposes of this project, I ended up cloning HF's repo for the APPS metric (https://huggingface.co/spaces/codeparrot/apps_metric) and tweaking it -- but please note that the modifications I made did not alter the robust testing framework functionality already in place. The two modifications I made were very unique to my needs:

- I wanted to leverage the code testing infrastructure afforded by the codebase for cross-validation, not just testing. All I had to do was change one line in the utils.py file to specify the "train" instead of the "test" split. (Note that, for this project, I opted to use the APPS train split as the development/validation/cross-validation set -- as the train split would have otherwise gone unused.)

- I wanted more granular information during eval, instead of just seeing the Pass@K summary at the end. For example, I wanted to see which questions specifically the model passes, and which out of the multiple code generations per question, pass. This was especially useful during cross-validation to compare prompt designs and also for error analysis for the paper. A related tweak was to display the running pass@5 score in real-time on the terminal during the course of evaluation -- it's just nice to see live performance results as the script progresses!

My clone with the above modifications is available here: https://github.com/N-G-Asker/apps_metric

The APPS dataset was introduced in the paper "Measuring Coding Challenge Competence With APPS" (https://arxiv.org/pdf/2105.09938.pdf) and is considered a difficult benchmark in the code-generation domain. Citation: 
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}

Usage: use as described in eval/eval-scripts/readme.md

"""
import json
import pickle
import pprint
from evaluate import load
import sys
import os

in_file = sys.argv[1]

DATA_PATH = "codeparrot/apps"

# setup
print("Loading evaluation dataset...")  
metric = load("apps_metric") # for offline use, after downloading from
                             # git clone https://huggingface.co/spaces/codeparrot/apps_metric
# metric = load("loubnabnl/apps_metric") # when online

with open(os.path.join(os.path.dirname(__file__), in_file), "rb") as fp1:   # Unpickling
    generations = pickle.load(fp1)

print(f"\nUnpickled input file {in_file}...\n")

results = metric.compute(predictions=generations,
                            level="introductory",
                            k_list=[1, 2, 3, 4, 5],
                            count_errors=False,
                            debug=False)
print(results)
output_file = f"pass_AT_k_metric_GPTQ_using_input_file{in_file}_"

with open(output_file, "w") as fp2:
    json.dump(results, fp2)