from timeit import default_timer as timer


def run_and_measure_time(func, kwargs, logger=None):
    start = timer()
    return_values = func(**kwargs)
    end = timer()
    elapsed_time = end - start
    if logger is not None:
        logger.debug(f"Elapsed time: {elapsed_time}")
    return return_values, elapsed_time
