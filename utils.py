import time


def log_execution_time(func=None, prefix=""):
    """
    A decorator that logs the execution time of a function.

    Args:
        func (callable, optional): The function to be decorated.
        prefix (str, optional): A prefix to include in the log message.

    Returns:
        callable: The decorated function.
    """

    def decorator(inner_func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = inner_func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            log_message = f"{prefix}{inner_func.__name__} took {execution_time:.4f} seconds to execute."
            print(log_message)

            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
