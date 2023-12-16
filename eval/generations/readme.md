# Generations
Each experiment has a subfolder with all artifacts of intermediate and final (code) generations, (which were saved by serializing them as python pickle objects). Access them using the utils provided it in the `utils` top level [directory](/utils). Unfortunately, The way these data files were named should be more or less disregarded

For convenience, printed versions of the underlying artifacts are provided, offering a clean viewing experience. These are stored in the following files inside each experiment's folder, depending on the experiment (for example, experiment 1 only has code-gen artifacts).

- code-gens.txt

- plans.txt

- feedback.txt

- rewrites.txt

- prompts-for-code.txt