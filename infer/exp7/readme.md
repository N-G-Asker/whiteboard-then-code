# Baseline 7
Iterated plan feedback-rewrite (x2), where the feedback comes from a different peer LLM in each iteration. Both the first and second peers $X_1$ and $X_2$ are prompted as experts, but their prompting differs to elicit different behaviors and focus on different aspects of the plan generations.

- $X_1$ assumes the persona of an "algorithms expert"

- $X_2$ assumes the persona of a "practical software engineer"

The same script that was used in 4A (in which feedback is solicited from the "practical software engineer" teammate) is run, but passing in the rewrites generated from 4B (in which the rewrites were based on single-round feedback from the "algorithms expert" teammate). 