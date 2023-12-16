"""
This script is a heavily adapted version of the example_script.py on HuggingFace's public repository for the APPS  Metric: 
https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/example_script.py

The APPS dataset was introduced in the paper "Measuring Coding Challenge Competence With APPS" (https://arxiv.org/pdf/2105.09938.pdf) and is considered a difficult benchmark in the code-generation domain. Citation: 
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}

Example Usage:

nohup python <script.py> --model_ckpt TheBloke/CodeLlama-7B-Instruct-GPTQ --num_tasks 10 --difficulty introd
uctory --n_samples 5 --temperature 0.6 < /dev/null > script.log 2>&1 &

"""

import json
import torch
import pickle
import pprint
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from evaluate import load
from time import localtime, strftime

def generate_prompt(sample):
    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"] 
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"] 
    except ValueError:
        fn_name = None
    
    # following exact zero-shot prompt used in Code LLama paper Figure 13: https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/
    if fn_name:
        QUESTION_GUIDE = "use the provided function signature"
    else:
        QUESTION_GUIDE = "read from and write to standard IO"

    starter_code_text = f'{starter_code}\n'  if starter_code else ""
    
    _input = f"""[INST] Write a python code to solve the following coding problem that obeys the constraints and
                passes the example test cases. The output code needs to {QUESTION_GUIDE}. Please wrap your code
                answer using ```:\n
                {sample["question"]}\n
                {starter_code_text}
                [/INST]"""
    # _input = f"""<s>[INST] Write a python code to solve the following coding problem that obeys the constraints and
    #             passes the example test cases. The output code needs to {QUESTION_GUIDE}. Please wrap your code
    #             answer using ```:\n
    #             {sample["question"]}\n
    #             {starter_code_text}
    #             [/INST]"""
    # _input = f"""Write a python code to solve the following coding problem that obeys the constraints and
    #         passes the example test cases. The output code needs to {QUESTION_GUIDE}. Please wrap your code
    #         answer using ```:\n
    #         {sample["question"]}\n
    #         {starter_code_text}
    #         """

    return _input


def complete_code(pipe, prompt, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    # prompt = pipe.tokenizer.eos_token + prompt
    try:
        code_gens = pipe(prompt, num_return_sequences=num_completions, **gen_kwargs)
        # return [code_gen["generated_text"] for code_gen in code_gens]
        return [code_gen["generated_text"][len(prompt):] for code_gen in code_gens]
    except IndexError:
        print("prompt is longer than the context size of the model, generation skipped")
        code_gens = ""
        return [""]


def make_generations(dataset, args, model, tokenizer):
    set_seed(args.seed)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype="auto", max_new_tokens=1024)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }

    # Generate completions for evaluation set
    n_tasks = args.num_tasks if args.num_tasks is not None else len(dataset)
    print(f"ntasks is {n_tasks}")

    round = 0
    generations = []
    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = generate_prompt(dataset[task]).strip()
        task_generations.extend(complete_code(pipe, prompt, num_completions=args.n_samples, **gen_kwargs))
        # generations.append([gen.replace(pipe.tokenizer.eos_token, "") for gen in task_generations])
        gens = []
        for gen in task_generations:
            gen.replace(pipe.tokenizer.eos_token, "")
            
            # # Extract code from generations, situated between code fence ```
            gen_split = gen.split("```")

            code = gen_split[1] if len(gen_split)>1 else gen_split[0]

            # check for presence of statement "if __name__ == '__main__':"
            code_split = code

            gens.append(code)

            # gens.append(gen)

        generations.append(gens)

        # Periodically save generations to a file (every 50 problems)
        if round > 0 and round % 50 == 0:
            filename = f"generations_btwn_iters_{round-50}_and_{round}__"
            filename += strftime("%d_%b_%Y_%I.%M%p", localtime())
            with open(filename, "wb") as gen_dump:
                pickle.dump(generations[round-50:round], gen_dump)

        round += 1

    # Save generations to a file
    filename = "generations_all"
    filename += strftime("%d_%b_%Y_%I.%M%p", localtime())
    with open(filename, "wb") as gen_dump:
        pickle.dump(generations, gen_dump)

    return generations


def main(args):
    DATA_PATH = "codeparrot/apps" # See the following for the full repository: https://huggingface.co/datasets/codeparrot/apps
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # setup
    print("Loading evaluation dataset...")
    dataset = load_dataset(DATA_PATH, split="test", difficulties=[args.difficulty])
    if args.use_solutions:
        print("Using data solutions as code generations")
        model = None
        tokenizer = None
        generations = []
        for index in range(args.num_tasks+1):
            try:
                sol = json.loads(dataset[index]["solutions"])
                generations.append(sol[:args.n_solutions])
            except ValueError:
                print(f"No solutions for task {index} or not enough to have {args.n_solutions} solutions")
                break
        
    else:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, use_fast=True)
        # https://huggingface.co/docs/accelerate/concept_guides/big_model_inference
        # model = AutoModelForCausalLM.from_pretrained(args.model_ckpt, device_map='cuda')
        model = AutoModelForCausalLM.from_pretrained(args.model_ckpt, device_map='auto')
        generations = make_generations(dataset, args, model, tokenizer)
        
    metric = load("loubnabnl/apps_metric")
    results = metric.compute(predictions=generations,
                             level=args.difficulty,
                             k_list=args.k_list,
                             count_errors=args.count_errors,
                             debug=args.debug)
    print(results)
    output_file = "pass_AT_k_metric_GPTQ"
    output_file += strftime("%d_%b_%Y_%I.%M%p", localtime())
    with open(output_file, "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing a Language Model on APPS Python Code dataset")
    #model and tokenizer arguments
    parser.add_argument("--model_ckpt", default="loubnabnl/apps-1.5B-model", type=str, help="path to model checkpoint.")
    parser.add_argument("--tokenizer", default="gpt2", type=str, help="tokenizer to use.")
    parser.add_argument("--eos", default="<|endoftext|>", type=str, help="end of sentence token.")
    # generation arguments
    parser.add_argument("--do_sample", default=True, type=bool, help="do sampling in generation")
    parser.add_argument("--temperature", default=0.2, type=float, help="temperature for sampling")
    parser.add_argument("--top_p", default=0.95, type=float, help="top p for sampling")
    parser.add_argument("--top_k", default=0, type=float, help="top k for sampling")
    parser.add_argument("--max_length", default=10000, type=int, help="max length of generated code")
    # evaluation arguments
    parser.add_argument("--difficulty", default="all", type=str, help="difficulty level to select in the dataset from:\
     'all', 'introductory', 'interview'  and 'competition' ")
    parser.add_argument("--num_tasks", default=6, type=int, help="number of tasks (questions in APPS) to generate on and evaluate")
    parser.add_argument("--use_solutions", default=False, type=bool, help="use solutions instead of generating new code")
    parser.add_argument("--n_samples", default=5, type=int, help="number of samples to generate, i.e., number of independent sequences to generate per task")
    parser.add_argument("--n_solutions", default=1, type=int, help="number of solutions to use")
    parser.add_argument("--k_list", default=[1, 2, 3, 4, 5], type=list, help="list of k values to evaluate pass@k")
    parser.add_argument("--count_errors", default=False, type=bool, help="count compilation and runtime errors for single generations")
    # configuration
    parser.add_argument("--seed", default=0, type=int, help="generation seed")
    # parser.add_argument("--device_int", default=-1, type=int, help="device on which code generation is run, if positive use GPU")
    # parser.add_argument("--device_int", default="cuda:0", type=str, help="device on which code generation is run, if positive use GPU")
    parser.add_argument("--debug", default=False, type=bool, help="debug mode")
    # save
    parser.add_argument("--output_file", default="apps_metrics.json", type=str, help="output file to save the results")

    args = parser.parse_args()
    main(args)