"""
Usage:
    python clean_generations.py <generations_pkl_file>
"""
import pickle
import sys
import os

in_file = sys.argv[1]

with open(os.path.join(os.path.dirname(__file__), in_file), "rb") as fp1:   # Unpickling
    codes = pickle.load(fp1)

for i, task in enumerate(codes):
    for j, attempt in enumerate(task):
        #############################################
        # For test framework compatibility,
        # find and reformat code block under idiom:
        #
        #     if __name__ == "__main__"
        #
        # In the docs this statement is referred to as a top-level 
        # code environment "guard":
        #     https://docs.python.org/3/library/__main__.html
        #############################################

        # find and split into three pieces if found
        partition = attempt.partition('if __name__ == \"__main__\":\n')
        
        if len(partition[2]) > 0:
            split = partition[2].split("\n")

            t = ''
            for s in split:
                t+= f"{s.removeprefix('    ')}\n"  # remove forerunning 4 spaces
            
            new = partition[0] + t

            codes[i][j] = new

        # Catch other variant. with single instead of double quotes around __main__
        # find and split into three pieces if found
        partition = attempt.partition('if __name__ == \'__main__\':\n')
        
        if len(partition[2]) > 0:
            split = partition[2].split("\n")

            t = ''
            for s in split:
                t+= f"{s.removeprefix('    ')}\n"  # remove forerunning 4 spaces
            
            new = partition[0] + t

            codes[i][j] = new

for i, task in enumerate(codes):
    for j, attempt in enumerate(task):

        #######################################################################
        # Remove open [PYTHON] and close [/PYTHON] tokens, as they cause
        # immedaite failure of tests. In these cases, the model followed the 
        # instructions in the prompt and wrapped its code in ```, but
        # additionally thought it necessary to indicate the language of the code
        # inside this fence.
        ########################################################################
        codes[i][j] = codes[i][j].replace("[PYTHON]", "")
        codes[i][j] = codes[i][j].replace("[/PYTHON]", "")

# Save generations to a file
filename = f"{in_file}_CLEANED.pkl"
with open(filename, "wb") as gen_dump:
    pickle.dump(codes, gen_dump)
