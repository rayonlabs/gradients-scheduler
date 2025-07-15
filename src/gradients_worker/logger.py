import contextvars
import logging

# Create a context-local variable to store the current task name
current_task = contextvars.ContextVar("current_task", default="main")


class TaskContextFilter(logging.Filter):
    def filter(self, record):
        record.task = current_task.get()
        return True


def setup_logger():
    logger = logging.getLogger("gradients_worker")
    logger.setLevel(logging.INFO)

    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s - %(task)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    task_filter = TaskContextFilter()
    handler.addFilter(task_filter)

    logger.addHandler(handler)
    return logger


logger = setup_logger()


class task_context:
    def __init__(self, task_name):
        self.task_name = task_name
        self.token = None

    def __enter__(self):
        self.token = current_task.set(self.task_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_task.reset(self.token)
