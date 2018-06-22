import numpy as np
from decimal import Decimal

def __get_leading_zeros(fl):
    """Returns the number of leading zeros in a float decimal."""
    if fl > 1.0:
        return 0
    else:
        fl_splitted = str(fl).split(".")[-1]
        N_unstripped = len(fl_splitted)
        N_left_stripped = len(fl_splitted.lstrip("0"))
        return N_unstripped - N_left_stripped

def __float_error_to_string(fl, prec):
    """Converts error to usable string."""
    num = "{:.15f}".format(fl)
    num = num.split(".")[-1]
    num = num[:prec]
    num = num.lstrip("0")
    return num

def sciprint(val, err, prec=0, force_prec=False):
    """
    Gets the error with uncertainty in scientific notation.

    Args:
        val: float, value to print error for.
        err: float, error in value.
        prec: int, optional. Default is 0, which means it will try to set the
            precision according to standard methods of writing errors.
        force_prec: bool, optional. If True, will force given precision. 
            Default is False.
    """
    val = Decimal(val)
    err = Decimal(err)

    # Splits error string into non-decimals and decimals
    val_leading, val_decimal = "{:.12f}".format(val).split(".")
    err_leading, err_decimal = "{:.12f}".format(err).split(".")

    # print val_decimal

    # How large the error are compared to the value
    error_magnitude = np.floor(float(np.log10(np.abs(val) / err)))

    # Number of leading zeros when we have err<0
    leading_error_zeros = __get_leading_zeros(err)

    if prec == 0:
        if error_magnitude >= 1:
            # If we have an order of magnitude or greater, we should only include 1 decimal
            prec = leading_error_zeros + 1
        else:
            # Same order of magnitude should have 2 extra decimals
            prec = leading_error_zeros + 2 

    max_err_prec = len(err_decimal.rstrip("0"))
    max_val_prec = len(val_decimal.rstrip("0"))

    # Checks if we have an provided error prec larger than what we can probe
    # And if it is larger than the maximum possible value precision
    if ((prec > max_err_prec and prec > max_val_prec) and not force_prec):
        prec = max_err_prec

    # Performs a round-off regardless of the error situation
    rounded_error = round(err, prec)

    if err < 1.0:
        # Converts float error to string while stripping off needed zeros
        str_err = __float_error_to_string(rounded_error, prec)
    else:
        # If we have two orders of magnitude larger value than error, we will
        # ignore decimals.
        if error_magnitude >= 2:
            prec = 0

        # No need to strip off leading zeros, so a simple string format is ok.
        str_err = "{:<.{w}f}".format(rounded_error, w=prec)

    # Converts value to string for a more symmetric view    
    str_val = "{0:<2.{w}f}".format(val, w=prec)
    
    msg = "{0:s}({1:s})".format(str_val, str_err)

    return msg


def main():
    def check_answer(name, val, err, ans, prec=0, force_prec=False):
        sciprint_ans = sciprint(val, err, prec=prec, force_prec=force_prec)
        unit_test = sciprint_ans==ans
        msg = "{:>3s}: ".format(name)
        msg += "{:12s} ".format(sciprint_ans)
        msg += "{:<2s} ".format(str(unit_test))
        if not unit_test:
            msg += "{:s} ".format(str(val))
            msg += "{:s} ".format(str(err))
            msg += "{:s} ".format(ans)
        print msg

    a = 0.2345
    a_err = 0.0049064
    a_ans = "0.234(5)"

    b = 0.111212
    b_err = 0.000084
    b_ans = "0.11121(8)"

    c = 0.111187
    c_err = 0.000160
    c_ans = "0.1112(2)"
    c_ans_prec = "0.11119(16)" # with prec=5

    d = 123.654
    d_err = 2.3
    d_ans = "123.7(2.3)"

    e = 12365.54
    e_err = 2.3
    e_ans = "12366(2)"

    f = 123.654
    f_err = 12.3
    f_ans = "123.7(12.3)"

    g = 2.7984
    g_err = 0.0009
    g_ans = "2.7984(9)"

    h = 2.995
    h_err = 0.004
    h_ans = "2.995(4)"

    # This one will be False, as they are not self consist in their notation
    i = 6.2191
    i_err = 0.0020
    i_ans = "6.2191(20)"

    # This one will require prec=2 to show full since magnitude is > 2
    j = 80.90
    j_err = 0.22
    j_ans = "80.90(22)"

    k = 0.81
    k_err = 0.11
    k_ans = "0.81(11)"

    m = 2.995
    m_err = 0.004
    m_ans = "2.995(4)"

    n = 0.1113
    n_err = 0.0010
    n_ans = "0.1113(10)"

    p = 0.1110
    p_err = 0.0010
    p_ans = "0.1110(10)"

    q = 28.14
    q_err = 0.14
    q_ans = "28.14(14)"

    r = 9.58
    r_err = 0.09
    r_ans = "9.58(9)"

    check_answer("a", a, a_err, a_ans)
    check_answer("b", b, b_err, b_ans)
    check_answer("c", b, c_err, c_ans)
    check_answer("cp", c, c_err, c_ans_prec, prec=5)
    
    # Error greater than 1.0
    check_answer("d", d, d_err, d_ans)
    check_answer("e", e, e_err, e_ans)
    check_answer("f", f, f_err, f_ans)

    check_answer("g", g, g_err, g_ans, prec=4)
    check_answer("h", h, h_err, h_ans, prec=3)

    check_answer("i", i, i_err, i_ans, prec=4)

    check_answer("j", j, j_err, j_ans, prec=2)

    check_answer("k", k, k_err, k_ans)
    check_answer("m", m, m_err, m_ans, prec=4)
    check_answer("n", n, n_err, n_ans, prec=4)
    check_answer("p", p, p_err, p_ans, prec=4, force_prec=True) # This cant be True and still be consistent with m

    check_answer("q", q, q_err, q_ans, prec=3)
    check_answer("r", r, r_err, r_ans, prec=3) # This cant be True and still be consistent with m

if __name__ == '__main__':
    main()