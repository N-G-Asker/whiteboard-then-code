# Testing with the Evaluation Scripts

This document is broken into three sections that overview the following:

1. __Code Execution Sandbox__: details on the setup of the secure evaluation infrastructure used

2. __Testing Code Generations How-To__: provides instructions for evaluating model outputs

3. __Tweaks Made to HuggingFace's APPS Metric Repo__: Outlines project-specific tweaks I made to HuggingFace's off-the-shelf APPS metric repository

## 1. Code Execution Sandbox
Because executing LLM generated code poses a security risk and should be treated as untrusted, I built a sandbox following (The Codex paper/cite{}). In particular, I only ran generated code inside of a containerized runtime isolated with gVisor from the host computer (a GCP VM instance). Moreover, external networking was disabled on the container. For additional details, see Appendix A.

### 1.1. Setup Details:
I emulated the testing sandbox setup used in the Codex paper (/cite{}), described in Section 2.3. "Sandbox for Executing Generated Code". 

__Procedure__

1. For setting up and using gVisor (https://gvisor.dev/), I relied on the following resources:

    - gVisor Installation: https://gvisor.dev/docs/user_guide/install/

    - Running Containers: https://gvisor.dev/docs/user_guide/quick_start/docker/ 

2. I also disabled external networking for Docker containers: https://gvisor.dev/docs/user_guide/networking/.

Helpful Docker resources for this project included:

- Docker Installation: https://docs.docker.com/engine/install/debian/

- Build python Docker image: https://hub.docker.com/_/python/

For testing infrastructure, I used a Google Cloud Platform (GCP) Virtual Machine (VM) instance running Debian-11-bullseye and with 16GB RAM. Under this setup, evaluating on the pass@5 metric took roughly ten minutes for every 100 questions (that is, pass@5 could be calculated at a rate of approximately 10 questions per minute).

## 2. Testing Code Generations How-To

The below assumes the secure sandbox has already been established (e.g., by installing and setting up gVisor, and disabling network access for containers).

### 2.1. Create Test Image

1. `scp` or otherwise copy one of the two `*-docker-uploads` directories to the VM you are using for evaluation, depending on your use-case:

    - `TEST-docker-uploads` - if you are testing

    - `XVAL-docker-uploads` - if you are doing cross-validation (e.g., trying out a new prompt design)
    
    These folders conveniently bundle together the files needed for tests. 
    
    __CAUTION:__ Be very careful which directory you use. The Dockerfiles they contain are different and in particular evaluate against different data splits. The `TEST-docker-uploads/Dockerfile` is programmed to evaluate against the `test` split of the APPS dataset. On the other hand, the `XVAL-docker-uploads/Dockerfile` is programmed to evaluate against the `train` split of the APPS dataset, which this project decided to use as the development/validation/Cross-Validation ("xval") split.

2. On the eval machine, go to the directory in which you want to create and launch your test. It should contain two python scripts, an associated requirements file for `pip` to install the necessary packages, and a Dockerfile. 

    __Transfer the specific code-generation output file__ you wish to test (which the generation script will have spit out in pickled format, even if the filename does not explicitly have the .pkl extension), into this directory (through `scp` or otherwise).

    The directory should now look as follows:

        $ ls -1
        Dockerfile
        <generations_file_with_date.pkl>
        clean_generations.py
        grading_gens.py
        requirements.txt

3. Update the two lines in the Dockerfile with <generations_file_with_date.pkl>.

4. Create the container image we'll use for this test by running the following:

        $ docker build --no-cache -t <test_name> .

    The `--no-cache` option is not strictly necessary but is good practice to avoid mysterious bugs if you make any updates to the scripts or the Dockerfile.

    Building takes about ~30 seconds.

### 2.2. Run the Image
Instantiate the container by running:

    $ docker run --runtime=runsc --rm <test_name> 2>&1 | tee out.txt

The `--runtime=runsc` specification tells Docker to isolate the container from the host using the special `gVisor` runtime. (Obviously, a prerequisite for this is having installed `gVisor` -- see above.)

The trailing "pipe-and-tee" clause `| tee out.txt` ensures the results are saved to a file named `out.txt` in the working directory, in addition to being printed out on the active terminal (i.e., to `std-out`).

## 3. Tweaks Made to HuggingFace's APPS Metric Repo

I leverage HuggingFace's easy-to-use python API for benchmarking/metrics/datasets to access a (publically-available, open-source) repo on the HF Hub dedicated to the APPS metric (https://huggingface.co/spaces/codeparrot/apps_metric). It's ready to use out of the box, and works great for testing.

I ended up cloning this repo and tweaking it for the purposes of this project, but note that the modifications I made did not alter the robust testing framework functionality already in place. The two modifications I made were very unique to my needs:

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
