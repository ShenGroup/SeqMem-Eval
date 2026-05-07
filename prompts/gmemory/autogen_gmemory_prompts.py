from __future__ import annotations

from dataclasses import dataclass


# AutoGen role prompts
SOLVER_SYSTEM_PROMPT = (
    "You are a smart agent designed to solve problems."
)

GROUND_TRUTH_SYSTEM_PROMPT = (
    "You are an agent designed to assist the solver agent when the solver is stuck or "
    "repeating an incorrect approach. Provide a different and more reliable strategy "
    "to move toward a correct answer."
)


# Task solving format
TASK_SOLVE_WITH_INSIGHTS = """
## Successful Examples (Reference Cases)
Below are some examples of similar tasks that were successfully completed.
Please use these as references to guide your thinking and approach to the current task:

{few_shots}
---

## Your Own Past Successes (Execution Patterns)
Here are examples of successful execution processes you've previously used on similar tasks.
Pay attention to step-by-step procedures and strategies:

{memory_few_shots}
---

## Key Insights from Related Tasks
The following are insights gathered during execution of similar tasks.

{insights}
---

## Your Turn
Solve the following task:
{task_description}
"""

TASK_FORMAT = """
### Task description:
{task_description}

### Key steps:
{key_steps}

### Detailed trajectory:
{trajectory}
"""


# GMemory-style prompts
GENERATIVE_TASK_SYSTEM_PROMPT = "You score relevance between a successful case and an ongoing task."
GENERATIVE_TASK_USER_PROMPT = """
You will be given a successful case and an ongoing task.
Evaluate how helpful the successful case is for the ongoing task on a scale of 1-10.

Success Case:
{trajectory}

Ongoing task:
{query_scenario}

Score:
"""

EXTRACT_TRUE_TRAJ_SYSTEM_PROMPT = (
    "You extract key steps from a trajectory. Keep only critical, valid, and reusable steps."
)
EXTRACT_TRUE_TRAJ_USER_PROMPT = """
Task:
{task}

Trajectory:
{trajectory}

Output requirements:
- Keep only key steps present in the trajectory.
- Remove clearly invalid/no-op steps when obvious.
- Output a numbered list of concise steps.
"""

DETECT_MISTAKES_SYSTEM_PROMPT = (
    "You analyze a failed trajectory and identify the most likely reason for failure."
)
DETECT_MISTAKES_USER_PROMPT = """
Task:
{task}

Failed trajectory:
{trajectory}

Output one concise failure reason:
"""

FORMAT_RULES_OPERATION_TEMPLATE = """
<OPERATION> <RULE NUMBER>: <RULE>

Allowed operations:
AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Constraints:
- At most 4 operations.
- Each on its own line.
- Rules must be concise and generally applicable.
"""

CRITIQUE_COMPARE_RULES_SYSTEM_PROMPT = (
    "You derive reusable rules by comparing one successful trajectory and one failed trajectory."
)
CRITIQUE_COMPARE_RULES_USER_PROMPT = """
## Trial 1 (success)
Task:
{task1}
Trajectory:
{task1_trajectory}

## Trial 2 (failed)
Task:
{task2}
Trajectory:
{task2_trajectory}
Failure reason:
{fail_reason}

## Existing Rules
{existing_rules}

Update rules using operations:
""" + FORMAT_RULES_OPERATION_TEMPLATE

CRITIQUE_SUCCESS_RULES_SYSTEM_PROMPT = (
    "You improve a reusable rule set based on successful trajectories."
)
CRITIQUE_SUCCESS_RULES_USER_PROMPT = """
## Successful histories
{success_history}

## Existing Rules
{existing_rules}

Update rules using operations:
""" + FORMAT_RULES_OPERATION_TEMPLATE

MERGE_RULES_SYSTEM_PROMPT = (
    "You merge redundant or overlapping rules into a concise non-redundant list."
)
MERGE_RULES_USER_PROMPT = """
Current rules:
{current_rules}

Rewrite into at most {limited_number} concise rules.
Return only a numbered list.
"""

PROJECT_INSIGHTS_SYSTEM_PROMPT = (
    "Adapt general insights into role-specific personalized insights."
)
PROJECT_INSIGHTS_USER_PROMPT = """
Role:
{role}

General insights:
{insights}

Return as numbered list.
"""

PROJECT_INSIGHTS_WITH_TRAJ_SYSTEM_PROMPT = (
    "Adapt general insights into role-specific insights given a trajectory context."
)
PROJECT_INSIGHTS_WITH_TRAJ_USER_PROMPT = """
Trajectory:
{trajectory}

Role:
{role}

General insights:
{insights}

Return as numbered list.
"""


@dataclass(frozen=True)
class AutoGenGMemoryPrompts:
    solver_system_prompt: str = SOLVER_SYSTEM_PROMPT
    ground_truth_system_prompt: str = GROUND_TRUTH_SYSTEM_PROMPT
    task_solve_with_insights: str = TASK_SOLVE_WITH_INSIGHTS
    task_format: str = TASK_FORMAT
    generative_task_system_prompt: str = GENERATIVE_TASK_SYSTEM_PROMPT
    generative_task_user_prompt: str = GENERATIVE_TASK_USER_PROMPT
    extract_true_traj_system_prompt: str = EXTRACT_TRUE_TRAJ_SYSTEM_PROMPT
    extract_true_traj_user_prompt: str = EXTRACT_TRUE_TRAJ_USER_PROMPT
    detect_mistakes_system_prompt: str = DETECT_MISTAKES_SYSTEM_PROMPT
    detect_mistakes_user_prompt: str = DETECT_MISTAKES_USER_PROMPT
    critique_compare_rules_system_prompt: str = CRITIQUE_COMPARE_RULES_SYSTEM_PROMPT
    critique_compare_rules_user_prompt: str = CRITIQUE_COMPARE_RULES_USER_PROMPT
    critique_success_rules_system_prompt: str = CRITIQUE_SUCCESS_RULES_SYSTEM_PROMPT
    critique_success_rules_user_prompt: str = CRITIQUE_SUCCESS_RULES_USER_PROMPT
    merge_rules_system_prompt: str = MERGE_RULES_SYSTEM_PROMPT
    merge_rules_user_prompt: str = MERGE_RULES_USER_PROMPT
    project_insights_system_prompt: str = PROJECT_INSIGHTS_SYSTEM_PROMPT
    project_insights_user_prompt: str = PROJECT_INSIGHTS_USER_PROMPT
    project_insights_with_traj_system_prompt: str = PROJECT_INSIGHTS_WITH_TRAJ_SYSTEM_PROMPT
    project_insights_with_traj_user_prompt: str = PROJECT_INSIGHTS_WITH_TRAJ_USER_PROMPT


PROMPTS = AutoGenGMemoryPrompts()


def format_task_prompt_with_insights(
    few_shots: list[str],
    memory_few_shots: list[str],
    insights: list[str],
    task_description: str,
) -> str:
    insight_text = "\n".join(f"{idx}. {r}" for idx, r in enumerate(insights, start=1))
    if not insight_text.strip():
        insight_text = "(empty)"
    memory_text = "\n\n".join(
        f"Task {idx + 1}:\n{shot}" for idx, shot in enumerate(memory_few_shots)
    )
    if not memory_text.strip():
        memory_text = "(empty)"
    few_shot_text = "\n".join(str(s) for s in few_shots if str(s).strip())
    if not few_shot_text.strip():
        few_shot_text = "(none)"
    return PROMPTS.task_solve_with_insights.format(
        few_shots=few_shot_text,
        memory_few_shots=memory_text,
        insights=insight_text,
        task_description=task_description,
    ).strip()


def format_task_context(task_description: str, task_traj: str, key_steps: str = "") -> str:
    return PROMPTS.task_format.format(
        task_description=str(task_description or "").strip(),
        key_steps=str(key_steps or "").strip(),
        trajectory=str(task_traj or "").strip(),
    ).strip()


def parse_numbered_list(text: str) -> list[str]:
    import re

    pattern = r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)"
    items = re.findall(pattern, str(text or "").strip(), flags=re.DOTALL)
    return [item.strip() for item in items if item.strip()]
