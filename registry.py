from methods import METHOD_REGISTRY
from tasks import TASK_REGISTRY


def build_method(method_name, **kwargs):
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}")
    return METHOD_REGISTRY[method_name](**kwargs)


def load_tasks(task_names, per_task_kwargs=None, **kwargs):
    """Instantiate tasks. kwargs are broadcast to every task; per_task_kwargs
    is a dict name -> extra kwargs applied only to that task. Per-task kwargs
    win on conflict."""
    per_task_kwargs = per_task_kwargs or {}
    tasks = []
    for name in task_names:
        if name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {name}")
        merged = {**kwargs, **per_task_kwargs.get(name, {})}
        tasks.append(TASK_REGISTRY[name](**merged))
    return tasks
