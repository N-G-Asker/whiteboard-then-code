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
import pickle
import pprint
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from evaluate import load
from time import localtime, strftime

REQUEST_FEEDBACK = "Does the high-level solution above make sense and answer the problem? If not, briefly give one suggestion on how to improve the logic. Don't fix it, just give a single suggestion."

def create_prompt_for_content_plan(sample):
    """
    The design of this content-planning prompt template borrows from both of the following:

    - the prompt template used in the Code Llama paper (https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/). See Appendix E "Zero shot results on APPS", and Appendix G "Prompts" Figure 13 "Prompts used to evaluate Code Llama on APPS."

    - Both the simplicity, and the specific  wording of the negative instruction of the form "Don't {do something}, just {do something else}" for output control purposes, are inspired by theSelf-Refine paper (https://arxiv.org/pdf/2303.17651.pdf). See Figure 22: FEEDBACK prompt for Code Readability.
    """
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
    
    # The following "Let's think step by step" phrasing is based on the Chain of Thought prompting technique from the paper "Large Language Models are Zero-Shot Reasoners"(https://arxiv.org/pdf/2205.11916.pdf")
    _input = ("[INST] Explain how to solve the following coding problem. Please do not code, just describe a solution using comments. Provide enough detail that someone else can implement the solution later:\n"
             f"{sample['question']}\n"
             f"{starter_code_text}"
              "[/INST] # Let's think step by step.\n")

    return _input

def create_prompt_for_code(sample, content_plan):
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

    starter_code_text = f'{starter_code}\n' if starter_code else ""
    
    _input = ("[INST] Write a python code to solve the following coding problem "
              "that obeys the constraints and passes the example test cases. "
              f"The output code needs to {QUESTION_GUIDE}. Please wrap your code "
              "answer using ```:\n"
              f"{sample['question']}\n"
              f"-----High-level Solution----- {content_plan}"
              f"{starter_code_text}"
              "[/INST]")

    return _input

def create_prompt_for_feedback(sample, original_plan):
    """
    The design of this feedback prompt template is inspired by choices in the Self-Refine paper (https://arxiv.org/pdf/2303.17651.pdf). See Figure 22: FEEDBACK prompt for Code Readability.
    """
    """
    Args:
        sample: the current  question
        plan: the initial content plan, on which we are soliciting the generator
              for feedback
    """
    # Based on the Self-Refine paper's prompt template for Math Reasoning feedback
    # See Figure 32 in Appendix
    # https://arxiv.org/pdf/2303.17651.pdf
    # REQUEST_FEEDBACK = "There is an error in the high-level solution above "
    # "because of lack of understanding of the question. What is the error? To "
    # "find the error, go through each comment line by line, and "
    # "check if everything looks good."

    # Based on the Self-Refine paper's prompt template for Code readibilty feedback
    # See Figure 23 in Appendix
    # https://arxiv.org/pdf/2303.17651.pdf
    # REQUEST_FEEDBACK = "Can you give one suggestion to improve the high-level solution above? Don't fix it, just give a suggestion."
    # The above led to very similar suggestions for all prompts

    # Based on the Self-Refine paper's prompt template for Code readibilty feedback
    # See Figure 23 in Appendix
    # https://arxiv.org/pdf/2303.17651.pdf
    # REQUEST_FEEDBACK = "Does the high-level solution above make sense and answer the question? If not, give one suggestion on how to improve the logic. Don't fix it, just give a suggestion."

    # REQUEST_FEEDBACK = "Does the above algorithm make sense and answer the problem? Explain your reasoning."

    # REQUEST_FEEDBACK = "Does the high-level solution above make sense and answer the problem? If not, how can you improve the logic? Don't fix it, just make suggestions."


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
    
    # Take advantage of Code Llama's conversation/chat/dialogue ability
    _input = ("[INST] Explain how to solve the following coding problem. Please do not code, just describe a solution using comments. Provide enough detail that someone else can implement the solution later:\n"
             f"{sample['question']}\n"
             f"{starter_code_text}"
              "[/INST] "
              "{plan} </s>"
              f"<s>[INST] {REQUEST_FEEDBACK} [/INST] ")

    return _input

def create_prompt_for_revised_plan(sample, original_plan, feedback_text):
    """
    Args:
        sample: the current  question
        plan: the initial content plan, on which we are soliciting the generator
              for feedback
    """
    # Based on the Self-Refine paper's prompt template for Math Reasoning feedback
    # See Figure 32 in Appendix
    # https://arxiv.org/pdf/2303.17651.pdf
    # REQUEST_FEEDBACK = "There is an error in the high-level solution above "
    # "because of lack of understanding of the question. What is the error? To "
    # "find the error, go through each comment line by line, and "
    # "check if everything looks good."

    # Based on the Self-Refine paper's prompt template for Code readibilty feedback
    # See Figure 23 in Appendix
    # https://arxiv.org/pdf/2303.17651.pdf
    # REQUEST_FEEDBACK = "Can you give one suggestion to improve the high-level solution above? Don't fix it, just give a suggestion."
    # The above led to very similar suggestions for all prompts

    # Based on the Self-Refine paper's prompt template for Code readibilty feedback
    # See Figure 23 in Appendix
    # https://arxiv.org/pdf/2303.17651.pdf
    # REQUEST_FEEDBACK = "Does the high-level solution above make sense and answer the question? If not, give one suggestion on how to improve the logic. Don't fix it, just give a suggestion."


    # REQUEST_FEEDBACK = "Does the above algorithm make sense and answer the problem? Explain your reasoning."

    # REQUEST_FEEDBACK = "Does the high-level solution above make sense and answer the problem? If not, how can you improve the logic? Don't fix it, just make suggestions."

    REQUEST_REVISION = "Okay! Now rewrite the high-level solution."

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
    
    # Take advantage of Code Llama's conversation/chat/dialogue ability
    _input = ("[INST] Explain how to solve the following coding problem. Please do not code, just describe a solution using comments. Provide enough detail that someone else can implement the solution later:\n"
             f"{sample['question']}\n"
             f"{starter_code_text}"
              "[/INST] "
              "{plan} </s>"
              f"<s>[INST] {REQUEST_FEEDBACK} [/INST] "
              "{feedback_text} </s>"
              f"<s>[INST] {REQUEST_REVISION} [/INST] ")

    return _input

def complete_code(pipe, prompt, num_completions=1, **code_gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    # prompt = pipe.tokenizer.eos_token + prompt
    try:
        code_gens = pipe(prompt, num_return_sequences=num_completions, **code_gen_kwargs)
        # return [code_gen["generated_text"] for code_gen in code_gens]
        return [code_gen["generated_text"][len(prompt):] for code_gen in code_gens]
    except IndexError:
        print("prompt is longer than the context size of the model, generation skipped")
        code_gens = ""
        return [""]

def complete_plan(pipe, prompt, num_completions=1, **plan_gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    # prompt = pipe.tokenizer.eos_token + prompt
    try:
        prompt_and_plan_singleton = pipe(prompt, num_return_sequences=num_completions, **plan_gen_kwargs)
        return prompt_and_plan_singleton[0]["generated_text"][len(prompt):]
        
    except IndexError:
        print("prompt is longer than the context size of the model, generation skipped")
        return ""

def make_generations(dataset, args, model, tokenizer):
    set_seed(args.seed)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype="auto")

    n_tasks = args.num_tasks if args.num_tasks is not None else len(dataset)
    print(f"ntasks is {n_tasks}")

    # Load original plans C0
    with open(args.original_plans, "rb") as fp:   # Unpickling
        C0 = pickle.load(fp)

    ########################################################
    # Generate feedback, 1 for each plan correspoding to a problem (a.k.a "task")
    ########################################################

    # Feedback Generation settings
    feedback_gen_kwargs = {
        "do_sample": False, # for greedy decoding, following https://arxiv.org/pdf/2205.11916.pdf
        "max_new_tokens": 256 # same amount allowed for generating the initial plan
        }

    round = 0
    feedback = []
    for task in tqdm(range(n_tasks)):
        prompt = create_prompt_for_feedback(dataset[task], original_plan=C0[round]).strip()
        gen = complete_plan(pipe, prompt, num_completions=1, **feedback_gen_kwargs)
        gen.replace(pipe.tokenizer.eos_token, "")
        feedback.append(gen)

        # Periodically save generations to a file
        if round > 0 and round % 50 == 0:
            filename = f"b3_feedback_up_to_iter_{round}__"
            filename += strftime("%d_%b_%Y_%I.%M%p", localtime())
            with open(filename, "wb") as gen_dump:
                pickle.dump(feedback, gen_dump)

        round += 1

    # Save generations to a file
    filename = f"b3_feedback_all_{round}_iters_"
    filename += strftime("%d_%b_%Y_%I.%M%p", localtime())
    with open(filename, "wb") as gen_dump:
        pickle.dump(feedback, gen_dump)


    ########################################################
    # Generate new plans C_1
    ########################################################

    # Feedback Generation settings
    revision_gen_kwargs = {
        "do_sample": False, # for greedy decoding, following https://arxiv.org/pdf/2205.11916.pdf
        "max_new_tokens": 512 # double the amount allowed for generating the initial plan
        }

    round = 0
    revisions = []
    for task in tqdm(range(n_tasks)):
        prompt = create_prompt_for_revised_plan(dataset[task], original_plan=C0[round], feedback_text=feedback[round]).strip()
        gen = complete_plan(pipe, prompt, num_completions=1, **revision_gen_kwargs)
        gen.replace(pipe.tokenizer.eos_token, "")
        revisions.append(gen)

        # Periodically save generations to a file
        if round > 0 and round % 50 == 0:
            filename = f"b3_revisions_up_to_iter_{round}__"
            filename += strftime("%d_%b_%Y_%I.%M%p", localtime())
            with open(filename, "wb") as gen_dump:
                pickle.dump(revisions, gen_dump)

        round += 1

    # Save generations to a file
    filename = f"b3_revisions_all_{round}_iters_"
    filename += strftime("%d_%b_%Y_%I.%M%p", localtime())
    with open(filename, "wb") as gen_dump:
        pickle.dump(revisions, gen_dump)


    ###############################################################
    # Generate code, we make `n_samples` generations per task
    ###############################################################
    
    # Code Generation settings
    code_gen_kwargs = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": 1024
    }
    
    round = 0
    generations = []
    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = create_prompt_for_code(dataset[task], revisions[round]).strip()
        task_generations.extend(complete_code(pipe, prompt, num_completions=args.n_samples, **code_gen_kwargs))
        # generations.append([gen.replace(pipe.tokenizer.eos_token, "") for gen in task_generations])
        gens = []
        for gen in task_generations:
            gen.replace(pipe.tokenizer.eos_token, "")
            
            # # Extract code from generations, situated between code fence ```
            gen_split = gen.split("```")

            gens.append(gen_split[1] if len(gen_split)>1 else gen_split[0])

        generations.append(gens)

        # Periodically save generations to a file
        if round > 0 and round % 50 == 0:
            filename = f"b3_generations_up_to_iter_{round}__"
            filename += strftime("%d_%b_%Y_%I.%M%p", localtime())
            with open(filename, "wb") as gen_dump:
                pickle.dump(generations, gen_dump)

        round += 1

    # Save generations to a file
    filename = f"b3_generations_all_{round}_iters_"
    filename += strftime("%d_%b_%Y_%I.%M%p", localtime())
    with open(filename, "wb") as gen_dump:
        pickle.dump(generations, gen_dump)


def main(args):
    DATA_PATH = "codeparrot/apps" # See the following for the full repository: https://huggingface.co/datasets/codeparrot/apps
    argsdict = vars(args)
    # print(pprint.pformat(argsdict))

    # setup
    print("Loading evaluation dataset...")
    dataset = load_dataset(DATA_PATH, split="train", difficulties=[args.difficulty])

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, use_fast=True)
    # https://huggingface.co/docs/accelerate/concept_guides/big_model_inference
    # model = AutoModelForCausalLM.from_pretrained(args.model_ckpt, device_map='cuda')
    model = AutoModelForCausalLM.from_pretrained(args.model_ckpt, device_map='auto')
    make_generations(dataset, args, model, tokenizer)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing a Language Model on APPS Python Code dataset")
    #model and tokenizer arguments
    parser.add_argument("--model_ckpt", default="TheBloke/CodeLlama-7B-Instruct-GPTQ", type=str, help="path to model checkpoint.")
    # generation arguments
    parser.add_argument("--temperature", default=0.2, type=float, help="temperature for sampling")
    parser.add_argument("--top_p", default=0.95, type=float, help="top p for sampling")
    parser.add_argument("--top_k", default=0, type=float, help="top k for sampling")
    # evaluation arguments
    parser.add_argument("--difficulty", default="all", type=str, help="difficulty level to select in the dataset from:\
     'all', 'introductory', 'interview'  and 'competition' ")
    parser.add_argument("--num_tasks", default=6, type=int, help="number of tasks (questions in APPS) to generate on and evaluate")
    parser.add_argument("--n_samples", default=5, type=int, help="number of samples to generate, i.e., number of independent sequences to generate per task")
    parser.add_argument("--k_list", default=[1, 2, 3, 4, 5], type=list, help="list of k values to evaluate pass@k")
    # configuration
    parser.add_argument("--seed", default=0, type=int, help="generation seed")
    parser.add_argument("--original_plans", default="", type=str, help="path to pickle file containing the inital content plan generations.")
    args = parser.parse_args()
    main(args)