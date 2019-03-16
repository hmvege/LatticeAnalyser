#!/usr/bin/env python2
def timing_function(func):
    """
    Time function.

    Will time method if 'timefunc==True' is provided in function
    """
    def wrapper(*args, **kwargs):
        if "timefunc" in kwargs and kwargs["timefunc"] == True:
            import time
            timefunc = True
            kwargs.pop("timefunc")
        else:
            timefunc = False

        if timefunc:
            t1 = time.clock()

        val = func(*args, **kwargs)

        if timefunc:
            t2 = time.clock()

            time_used = t2 - t1

            print("Time used with function %s: %.10f secs/"
                  " %.10f minutes" % (func.__name__, time_used, time_used/60.))

        return val

    return wrapper
