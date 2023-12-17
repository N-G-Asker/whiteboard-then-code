# Whiteboard-Then-Code

## Read the Paper
This framework is the culmination of a semester-long independent project for the course _Natural Language Generation and Summarization_ (Fall 2023), a graduate seminar in the Computer Science department at Columbia University taught by Professor Kathleen McKeown. Read the (unpublished) research paper [here](whiteboard-then-code_research-paper.pdf).

Abstract:
>Although state-of-the-art large language models (LLMs) match the performance of human programmers on some coding tasks, they still struggle on difficult programming problems. To improve model performance on coding challenges, this work explores initially creating a content plan $-$ Whiteboarding $-$ before generating code, on the hypothesis that encouraging the generator to think deeply about a high-level solution could help overcome flawed reasoning. Further, it looks at how performance changes when the generator is allowed to iteratively improve the quality of its whiteboarded content plans through Self-Refine. These results are then compared to Team-Refine, a novel extension of Self-Refine intended to simulate human collaboration. In particular, it proposes that distinct peer LLMs, or teammates, provide the feedback instead of the generator itself during the Whiteboard phase, with each teammate bringing a unique background or area of expertise (e.g., efficiency, readability, correctness) to the table. In light of the suggestions, the generator is asked to adjust its plan and then implement the solution in code, using the finalized plan as a blueprint. This study focuses on the effects of collaboration applied during planning (the Whiteboard phase), leaving study of how collaboration fares at other points in the process, such as at code-implementation time (the Code phase), to future work.

## Repository Overview

There are three top-level directories in this project:

- `eval`: Contains the scripts and [docs](eval/eval-scripts/readme.md) for testing, as well as experiment outputs (generations) and results

- `infer`: Contains the generation scripts for all experiments

- `utils`: Contains handy scripts for viewing generations and byproducts (plans, feedback, rewrites, and prompts)

Drill down to each one and find their respective `readme`s for more details.

- [eval](/eval/readme.md)

- [infer](/infer/readme.md)

- [utils](/utils/readme.md)