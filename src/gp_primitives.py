"""This module contains primitive functions for GP primitive set."""
import numpy as np

def if_then_else(cond, o1, o2):
    """If then else function

    Args:
        cond (bool): Condition.
        o1 (any): Return element 1.
        o2 (any): Return element 2.

    Returns:
        (any): o1 if cond is true else o2.
    """
    return o1 if cond else o2

def if3(cond1, cond2, a, b, c):
    """If elif else function

    Args:
        cond1 (bool): Condition 1.
        cond2 (bool): Condition 2.
        o1 (any): Return element 1.
        o2 (any): Return element 2.
        o3 (any): Return element 3.

    Returns:
        (any): o1 if cond1 is true else o2 if cond2 is true else o3.
    """
    if cond1:
        return a
    if cond2:
        return b
    return c

def gt0(x):
    """Greater than zero function.

    Args:
        x (numeric): Number to compare with zero.

    Returns:
        bool: If x > 0.
    """
    return x > 0

def to_float(x):
    """Convert number to float.

    Args:
        x (numeric): Number for conversion.

    Returns:
        float: Converted number.
    """
    return float(x)

def to_int(x):
    """Convert number to int.

    Args:
        x (numeric): Number for conversion.

    Returns:
        int: Converted number.
    """
    return int(x)

def protected_div(a, b):
    """Protected division against zero division.

    Args:
        a (numeric): Dividend.
        b (numeric): Divisor.

    Returns:
        float: a / b if b is not zero else 1.0.
    """
    return 1.0 if b == 0 else a / b

def protected_sqrt(x):
    """Protected square function against negative x.

    Args:
        x (numeric): Number for sqrt.

    Returns:
        float: sqrt(abs(x)).
    """
    return np.sqrt(abs(x))

def protected_log(x):
    """Protected log function against negative and zero x.

    Args:
        x (numeric): Number for log.

    Returns:
        float: log(abs(x) + 1e-6).
    """
    return np.log(abs(x) + 1e-6)
