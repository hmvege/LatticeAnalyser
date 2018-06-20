import numpy as np
from decimal import Decimal

def __get_leading_zeros(fl):
    """Returns the number of leading zeros in a decimal."""
    if fl > 1.0:
        return 0
    else:
        fl_splitted = str(fl).split(".")[-1]
        N_unstripped = len(fl_splitted)
        N_left_stripped = len(fl_splitted.lstrip("0"))
        return N_unstripped - N_left_stripped

def __round_error(fl, round_off=0):
    """
    Returns error terms leading digit automagically.

    Only works for errors < 0.
    """
    fl = Decimal(fl)
    N = __get_leading_zeros(fl)
    # assert N > 0, "error is larger than 0: %s" % fl
    if N > 0:
        fl = fl * Decimal(10**(N + 1.0))

    return round(fl, round_off)

def error_str(val, err, prec=0):
    """Gets the error with uncertainty in scientific notation."""
    val = Decimal(val)
    err = Decimal(err)
    N_err = __get_leading_zeros(err)

    err_magnitude = np.floor(float(np.log10(val / err)))

    if err >= 1.0:
        if err_magnitude >= 2:
            err_rounded = "{0:<d}".format(int(__round_error(err, round_off=0)))
            N_err = -1
        else:
            err_rounded = str(__round_error(err, round_off=1))
    else:
        err_rounded = "{0:<d}".format(int(__round_error(err, round_off=0)))

    if prec == 0:
        prec = N_err + 1

    # print N_err
    msg = "{0:<2.{werr}f}({1:s})".format(val, err_rounded, werr=prec)

    return msg



def main():
    a = 0.2345
    a_err = 0.0049064
    # 0.234(5)

    b = 0.111212
    b_err = 0.000084
    # 0.11121(8)

    c = 0.111187
    c_err = 0.000160
    # 0.1112(2)

    d = 123.654
    d_err = 2.3
    # 123.7(2.3)

    e = 12365.54
    e_err = 2.3
    # 12366(2)

    f = 123.654
    f_err = 12.3
    # 123.7(12.3)

    # print "a:", error_str(a, a_err)

    print "b:", error_str(b, b_err)

    print "c:", error_str(c, c_err)

    # print "d:", error_str(d, d_err)

    # print "e:", error_str(e, e_err)

    # print "f:", error_str(f, f_err)

if __name__ == '__main__':
    main()