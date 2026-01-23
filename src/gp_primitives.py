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
    cond = np.asarray(cond, dtype=bool)
    o1 = np.asarray(o1)
    o2 = np.asarray(o2)
    return np.where(cond, o1, o2)

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
    cond1 = np.asarray(cond1, dtype=bool)
    cond2 = np.asarray(cond2, dtype=bool)
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    return np.select([cond1, cond2], [a, b], default=c)

def gt0(x):
    """Greater than zero function.

    Args:
        x (numeric): Number to compare with zero.

    Returns:
        bool: If x > 0.
    """
    return np.asarray(x, dtype=float) > 0

def to_float(x):
    """Convert number to float.

    Args:
        x (numeric): Number for conversion.

    Returns:
        float: Converted number.
    """
    return np.asarray(x, dtype=float)

def to_int(x):
    """Convert number to int.

    Args:
        x (numeric): Number for conversion.

    Returns:
        int: Converted number.
    """
    return np.asarray(x, dtype=int)

def protected_div(a, b):
    """Protected division against zero division.

    Args:
        a (numeric): Dividend.
        b (numeric): Divisor.

    Returns:
        float: a / b if b is not zero else 1.0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Suppress numpy warnings for divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
    return np.where(np.abs(b) < 1e-10, 1.0, result)

def protected_sqrt(x):
    """Protected square function against negative x.

    Args:
        x (numeric): Number for sqrt.

    Returns:
        float: sqrt(abs(x)).
    """
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 0.0, np.sqrt(np.abs(x)))

def protected_log(x):
    """Protected log function against negative and zero x."""
    x = np.asarray(x, dtype=float)
    safe = np.clip(np.abs(x), 1e-10, None)  # avoid log(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(safe)
    return np.where(np.isnan(result) | np.isinf(result), 0.0, result)

def protected_pow(x, y):
    """Protected power function."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_clamped = np.clip(y, -3, 3)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.power(np.abs(x), y_clamped)
    return np.where(np.isnan(result) | np.isinf(result), 1.0, result)

def protected_exp(x):
    """Protected exponential function."""
    x = np.asarray(x, dtype=float)
    x_clamped = np.clip(x, -50, 50)  # Prevent overflow
    with np.errstate(over='ignore'):
        result = np.exp(x_clamped)
    return np.where(np.isinf(result), 1e100, result)

def protected_reciprocal(x):
    """Protected 1/x function."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(1.0, x, out=np.ones_like(x, dtype=float), where=np.abs(x) >= 1e-10)
    return np.where(np.isnan(result) | np.isinf(result), 1.0, result)

def protected_not(x):
    """Protected boolean not."""
    x = np.asarray(x, dtype=bool)
    return ~x
