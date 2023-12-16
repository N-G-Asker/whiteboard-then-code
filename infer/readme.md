# Inference
This directory contains the scripts used for generating with the model.

Each subdirectory corresponding to a particular experiment also has a short `readme.md` describing it.

## How To Use the Scripts

On the virtual machine you're using for inference, install the necessary dependencies specified in the requirements.txt file.

    pip install -r requirements.txt

Navigate to the project directory and activate the python virtual environment.

    source venv/bin/activate


Use `nohup` command-line utility to detach from current SSH shell session, so that script continues even if connection drops. std_out and std_err is logged to specified file `script.log`. For details, see Michael Dillon's answer [here](https://unix.stackexchange.com/questions/30400/execute-remote-commands-completely-detaching-from-the-ssh-connection).

One-liner example:

    nohup python exp<>-script.py --model_ckpt TheBloke/CodeLlama-7B-Instruct-GPTQ --num_tasks 650 --difficulty introductory --n_samples 5 --temperature 0.6 </dev/null >script.log 2>&1 &


For later experiments, the scripts take an additional parameter to specify the old plans slated for refinement.

## Inference Machine Specs (GPUs)
For GPUs, I used two T4s -- a single T4's RAM was insufficient due to the large model size and long contexts in this project (which required up to approximately 20 GB of volatile RAM at points). The scripts are written to distribute load across the GPUs, taking advantage of HuggingFace's API and under-the-hood libraries like PyTorch to handle these low-level details automatically.