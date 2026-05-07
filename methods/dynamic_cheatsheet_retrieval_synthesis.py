import math
from pathlib import Path

from methods.base_method import BaseMethod
from utils.code_exec import (
    extract_last_python_block,
    format_exec_result,
    has_exec_directive,
    run_python,
)
from utils.llm_client import generate as llm_generate

try:
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except ImportError:
    transformers_logging = None

DEFAULT_GENERATOR_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "dc_rs"
    / "generator_prompt_official.txt"
)
DEFAULT_CURATOR_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "dc_rs"
    / "curator_prompt_for_dc_retrieval_synthesis_official.txt"
)


class DynamicCheatsheetRetrievalSynthesis(BaseMethod):
    name = "DC-RS"

    def __init__(
        self,
        top_k=3,
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        curator_model_name=None,
        temperature=0.0,
        max_new_tokens=1024,
        cheatsheet_default="(empty)",
        generator_prompt_path=None,
        curator_prompt_path=None,
        max_exec_rounds=3,
        exec_timeout_seconds=15,
        **kwargs,
    ):
        super().__init__()
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.curator_model_name = curator_model_name or generation_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.cheatsheet_default = cheatsheet_default
        self.generator_prompt_path = Path(generator_prompt_path or DEFAULT_GENERATOR_PROMPT_PATH)
        self.curator_prompt_path = Path(curator_prompt_path or DEFAULT_CURATOR_PROMPT_PATH)
        self.max_exec_rounds = max(0, int(max_exec_rounds))
        self.exec_timeout_seconds = max(1, int(exec_timeout_seconds))
        self._generator_prompt_template = self._read_prompt_template(
            self.generator_prompt_path, "generator"
        )
        self._curator_prompt_template = self._read_prompt_template(
            self.curator_prompt_path, "curator"
        )

        self._embedder = None
        self._task_cheatsheets = {}
        self._pending_cheatsheet = {}
        self._pending_debug = {}
        self._task_questions = {}
        self._task_question_embeddings = {}
        self._task_outputs = {}

    def _read_prompt_template(self, path, template_name):
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(
                f"DC-RS failed to read {template_name} prompt template: {path}"
            ) from exc

    def _render_template(self, template, replacements):
        rendered = str(template)
        for key, value in replacements.items():
            rendered = rendered.replace(key, str(value))
        return rendered

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from utils.embedder import get_embedder
        except ImportError as exc:
            raise RuntimeError(
                "DynamicCheatsheet_RetrievalSynthesis requires sentence-transformers. "
                "Please install it first: pip install sentence-transformers"
            ) from exc
        self._embedder = get_embedder(self.embedding_model_name)
        return self._embedder

    def _encode(self, texts):
        if not texts:
            return []
        embedder = self._load_embedder()
        vectors = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(vec) for vec in vectors]

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _extract_tag_content(self, text, tag_name):
        if not text:
            return ""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"
        if start_tag not in text or end_tag not in text:
            return ""
        try:
            content = text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
            return content
        except Exception:
            return ""

    def _build_retrieved_pairs_block(self, retrieved):
        if not retrieved:
            return "(empty)"
        blocks = [
            "### PREVIOUS SOLUTIONS (START)",
            "",
            "Note: The input-output pairs listed below are taken from previous test cases and are meant to assist you in understanding potential solution strategies or tool usages. While they can offer insight and inspiration, they should not be blindly copied, as they may contain errors or may not fit your specific use case. Approach them with a critical mindset-analyze their logic, verify their correctness, and adapt them as needed. Your goal should be to develop a well-reasoned solution that best addresses the problem at hand.",
            "",
        ]
        # Keep the same display order as the original implementation.
        for idx, item in enumerate(retrieved[::-1], start=1):
            question = item.get("question", "")
            model_output = item.get("model_output", "")
            score = item.get("retrieval_score")
            score_text = f" (Similarity: {score:.2f})" if isinstance(score, float) else ""
            blocks.append(f"#### Previous Input #{idx}{score_text}:")
            blocks.append(str(question))
            blocks.append("")
            blocks.append(f"#### Model Solution to Previous Input #{idx}:")
            blocks.append(str(model_output))
            blocks.append("---")
            blocks.append("---")
            blocks.append("")
        blocks.append("#### PREVIOUS SOLUTIONS (END)")
        return "\n".join(blocks).strip()

    def _build_curator_prompt(self, previous_cheatsheet, retrieved_pairs, next_input):
        return self._render_template(
            self._curator_prompt_template,
            {
                "[[PREVIOUS_CHEATSHEET]]": previous_cheatsheet,
                "[[PREVIOUS_INPUT_OUTPUT_PAIRS]]": retrieved_pairs,
                "[[NEXT_INPUT]]": next_input,
            },
        )

    def _build_generator_prompt(self, prompt, cheatsheet):
        return self._render_template(
            self._generator_prompt_template,
            {
                "[[QUESTION]]": prompt,
                "[[CHEATSHEET]]": cheatsheet,
            },
        )

    def retrieve_memory(self, task, **kwargs):
        query = str(kwargs.get("query", "")).strip()
        if self.top_k <= 0:
            return []
        task_name = task.name
        questions = self._task_questions.get(task_name, [])
        outputs = self._task_outputs.get(task_name, [])
        embeddings = self._task_question_embeddings.get(task_name, [])
        if not questions or not outputs or not embeddings:
            return []
        if not query:
            import os as _os
            if _os.environ.get("BWT_DISABLE_RECENCY_FALLBACK"):
                return []
            start = max(0, len(questions) - self.top_k)
            return [
                {
                    "question": questions[i],
                    "model_output": outputs[i],
                    "retrieval_score": None,
                }
                for i in range(start, len(questions))
            ]

        query_vec = self._encode([query])[0]
        similarities = [self._cosine(query_vec, emb) for emb in embeddings]
        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )[: self.top_k]
        return [
            {
                "question": questions[i],
                "model_output": outputs[i],
                "retrieval_score": float(similarities[i]),
            }
            for i in top_indices
        ]

    def export_eval_context(self, task, entry, memory):
        return {
            "cheatsheet_before": self._task_cheatsheets.get(
                task.name, self.cheatsheet_default
            ),
        }

    def generate(self, task, prompt, **kwargs):
        retrieved = kwargs.get("memory", [])
        previous_cheatsheet = self._task_cheatsheets.get(task.name, self.cheatsheet_default)
        retrieved_pairs = self._build_retrieved_pairs_block(retrieved)

        curator_prompt = self._build_curator_prompt(
            previous_cheatsheet=previous_cheatsheet,
            retrieved_pairs=retrieved_pairs,
            next_input=prompt,
        )
        curator_output = llm_generate(
            model_name=self.curator_model_name,
            prompt=curator_prompt,
            temperature=self.temperature,
            max_new_tokens=max(self.max_new_tokens, 512),
        )

        curator_output = str(curator_output or "")
        synthesized_cheatsheet = self._extract_tag_content(curator_output, "cheatsheet")
        used_fallback = False
        if not synthesized_cheatsheet:
            synthesized_cheatsheet = retrieved_pairs
            used_fallback = True

        generation_prompt = self._build_generator_prompt(prompt, synthesized_cheatsheet)

        transcript_parts = []
        exec_trace = []
        current_prompt = generation_prompt
        rounds_used = 0

        for round_idx in range(self.max_exec_rounds + 1):
            rounds_used = round_idx + 1
            output = llm_generate(
                model_name=self.generation_model_name,
                prompt=current_prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            output = str(output or "")
            transcript_parts.append(output)

            if not has_exec_directive(output):
                break
            code = extract_last_python_block(output)
            if not code:
                exec_trace.append({
                    "round": round_idx,
                    "code": "",
                    "stdout": "",
                    "stderr": "(no python code block found despite EXECUTE CODE! directive)",
                    "timed_out": False,
                    "returncode": -1,
                })
                break
            if round_idx >= self.max_exec_rounds:
                # Budget exhausted: force one final no-code completion.
                final_prompt = (
                    current_prompt
                    + "\n\n[Your previous response]:\n"
                    + output
                    + "\n\n[Execution budget exhausted. Do NOT write any more code. "
                    "Provide the final answer now in the exact format the question requested.]"
                )
                forced = llm_generate(
                    model_name=self.generation_model_name,
                    prompt=final_prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                )
                transcript_parts.append(str(forced or ""))
                rounds_used += 1
                break

            exec_result = run_python(code, timeout=self.exec_timeout_seconds)
            exec_trace.append({
                "round": round_idx,
                "code": code,
                **exec_result,
            })
            exec_result_text = format_exec_result(exec_result)
            current_prompt = (
                current_prompt
                + "\n\n[Your previous response]:\n"
                + output
                + "\n\n"
                + exec_result_text
                + "\n\n[Continue. If the execution result solves the problem, "
                "provide the final answer now in the format the question requested "
                "(no more code). Otherwise, revise your approach.]"
            )

        final_output = "\n\n---\n\n".join(transcript_parts)

        self._pending_cheatsheet[task.name] = synthesized_cheatsheet
        self._pending_debug[task.name] = {
            "retrieved_count": len(retrieved),
            "has_cheatsheet_start_tag": "<cheatsheet>" in curator_output,
            "has_cheatsheet_end_tag": "</cheatsheet>" in curator_output,
            "used_fallback": used_fallback,
            "previous_cheatsheet": previous_cheatsheet,
            "curator_output_raw": curator_output,
            "extracted_cheatsheet": self._extract_tag_content(curator_output, "cheatsheet"),
            "final_cheatsheet_used": synthesized_cheatsheet,
            "exec_rounds_used": rounds_used,
            "exec_trace": exec_trace,
        }
        return final_output

    def update_memory(self, task, output, **kwargs):
        record = kwargs.get("record")
        if not record:
            return

        current_cheatsheet = self._pending_cheatsheet.get(
            task.name, self._task_cheatsheets.get(task.name, self.cheatsheet_default)
        )
        debug_payload = self._pending_debug.pop(task.name, None)
        self._task_cheatsheets[task.name] = current_cheatsheet

        question = str(record.get("question", "")).strip()
        model_output = str(record.get("model_output", str(output)))

        if task.name not in self._task_questions:
            self._task_questions[task.name] = []
            self._task_question_embeddings[task.name] = []
            self._task_outputs[task.name] = []
        self._task_questions[task.name].append(question)
        self._task_outputs[task.name].append(model_output)
        self._task_question_embeddings[task.name].extend(self._encode([question]))

        enriched = {
            "task": task.name,
            **record,
            "memory": current_cheatsheet,
        }
        if debug_payload is not None:
            enriched["memory_debug"] = debug_payload
        self.memory.append(enriched)

    def reset_task_state(self, task):
        self._task_cheatsheets.setdefault(task.name, self.cheatsheet_default)
        self._task_questions.setdefault(task.name, [])
        self._task_question_embeddings.setdefault(task.name, [])
        self._task_outputs.setdefault(task.name, [])
        return None

    def restore_state_from_memory(self, task):
        task_name = task.name
        task_records = [r for r in self.memory if r.get("task") == task_name]
        if not task_records:
            return
        # Restore cheatsheet from the last record's "memory" field.
        last_cheatsheet = task_records[-1].get("memory", self.cheatsheet_default)
        self._task_cheatsheets[task_name] = last_cheatsheet
        # Restore questions and outputs lists.
        questions = []
        outputs = []
        for rec in task_records:
            q = str(rec.get("question", "")).strip()
            o = str(rec.get("model_output", "")).strip()
            if q:
                questions.append(q)
                outputs.append(o)
        self._task_questions[task_name] = questions
        self._task_outputs[task_name] = outputs
        # Re-embed all questions so retrieval works.
        self._task_question_embeddings[task_name] = self._encode(questions)
        print(
            f"[DC-RS Resume] Restored {len(questions)} Q/A pairs, "
            f"cheatsheet len={len(last_cheatsheet)}"
        )
