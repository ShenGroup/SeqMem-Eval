from abc import ABC, abstractmethod


class BaseTask(ABC):
    name = "BaseTask"
    prompt_key = ""

    def iter_entries(self):
        # Default single-step behavior for simple tasks.
        yield None

    def build_prompt(self, entry):
        return self.get_prompt()

    def build_inputs(self, entry):
        return self.get_inputs()

    def evaluate_entry(self, output, entry):
        return self.evaluate(output)

    def get_query(self, entry, inputs):
        return str(inputs.get("question", ""))

    def build_memory_record(self, entry, output, feedback, score):
        return {
            "output": output,
            "feedback": feedback,
            "score": score,
        }

    def total_entries(self):
        return None

    def memory_filename(self, method_name):
        return f"{self.name}_{method_name}_memory.jsonl"

    @abstractmethod
    def get_prompt(self):
        raise NotImplementedError

    @abstractmethod
    def get_inputs(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, output):
        raise NotImplementedError
