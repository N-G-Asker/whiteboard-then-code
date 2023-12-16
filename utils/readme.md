# utils
There are three utilities, all with the same usage.

Usage:
    python <script.py> <target_file_with_generations_or_prompts>


- `show_written_generations.py` can be used to view the following types of generations:

    - plans

    - feedback

    - rewrites (that is, refined plans)
    
    This works out because all three types have the same underlying serialization structure (in particular, the generation scripts nest all of them in the same manner in python lists before preserving them as a pickle, so when they're deserialized, they can be accessed in the same way). From a codebase maintainability perspective, using the same utility for these use cases is preferred over creating separate copies with names like `show_plan_generations.py`, `show_feedback_generations.py`, and `view_rewrite_generations.py`.

- `show_code_generation.py` prints the coding outputs from the generator.

- `show_prompts.py` prints prompts. It takes as input some prompts file, which the generation scripts output while generating. They are helpful for debugging, reporting, and analysis, since they include the question and content-plans and/or feedback -- it can be very useful to view many of the pieces/moving parts simultaneously.