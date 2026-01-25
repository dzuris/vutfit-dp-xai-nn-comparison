"""Protected primitive functions for genetic programming.

Provides a set of robust mathematical and logical operations designed for use in
DEAP genetic programming primitive sets. Each function handles edge cases
(division by zero, logarithm of negative numbers, overflow) gracefully to
prevent runtime errors during GP evolution and tree evaluation.

Functions:
    Conditional operations:
        if_then_else(cond, o1, o2): Ternary conditional (if-else)
        if3(cond1, cond2, a, b, c): Three-way conditional (if-elif-else)
        gt0(x): Test if value is greater than zero
        protected_not(x): Boolean negation

    Protected arithmetic:
        protected_div(a, b): Division with zero-divisor protection
        protected_sqrt(x): Square root of absolute value
        protected_log(x): Natural logarithm with domain protection
        protected_pow(x, y): Power function with clamped exponent
        protected_exp(x): Exponential with overflow protection
        protected_reciprocal(x): 1/x with zero protection

Notes:
    - All functions accept and return numpy arrays (vectorized)
    - Protected functions return safe defaults instead of NaN/Inf
    - Used by GPModel to build robust symbolic regression trees

See Also:
    src.gp_model.GeneticProgrammingModel: Uses these primitives in DEAP primitive set
    deap.gp.PrimitiveSet: Framework for defining GP operations
"""
import numpy as np

def if_then_else(cond, o1, o2):
    """Ternary conditional operation (vectorized).

    Evaluates condition and returns first or second operand element-wise.

    Args:
        cond (array-like): Boolean condition(s).
        o1 (array-like): Values returned where cond is True.
        o2 (array-like): Values returned where cond is False.

    Returns:
        np.ndarray: Array with o1 where cond is True, o2 otherwise.

    Example:
        >>> if_then_else([True, False, True], [10, 20, 30], [5, 15, 25])
        array([10, 15, 30])
    """
    cond = np.asarray(cond, dtype=bool)
    o1 = np.asarray(o1)
    o2 = np.asarray(o2)
    return np.where(cond, o1, o2)

def if3(cond1, cond2, a, b, c):
    """Three-way conditional operation (if-elif-else, vectorized).

    Evaluates two conditions sequentially and returns corresponding values.

    Args:
        cond1 (array-like): First boolean condition.
        cond2 (array-like): Second boolean condition (checked if cond1 is False).
        a (array-like): Values returned where cond1 is True.
        b (array-like): Values returned where cond2 is True (and cond1 is False).
        c (array-like): Default values (where both conditions are False).

    Returns:
        np.ndarray: Array with a where cond1, b where cond2, c otherwise.

    Example:
        >>> if3([True, False, False], [False, True, False], [1, 2, 3], [4, 5, 6], [7, 8, 9])
        array([1, 5, 9])
    """
    cond1 = np.asarray(cond1, dtype=bool)
    cond2 = np.asarray(cond2, dtype=bool)
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    return np.select([cond1, cond2], [a, b], default=c)

def gt0(x):
    """Test if values are greater than zero.

    Args:
        x (array-like): Numeric values to test.

    Returns:
        np.ndarray[bool]: Boolean array (True where x > 0).

    Example:
        >>> gt0([1.5, 0, -2.3, 0.01])
        array([True, False, False, True])
    """
    return np.asarray(x, dtype=float) > 0

def protected_div(a, b):
    """Division with zero-divisor protection.

    Computes a/b element-wise, returning 1.0 where |b| < 1e-10 to prevent
    division by zero errors.

    Args:
        a (array-like): Dividend(s).
        b (array-like): Divisor(s).

    Returns:
        np.ndarray: a/b where b is non-zero, 1.0 otherwise.

    Example:
        >>> protected_div([10, 5, 3], [2, 0, 1])
        array([5.0, 1.0, 3.0])

    Notes:
        - Threshold 1e-10 prevents near-zero division instability
        - Returns 1.0 (neutral multiplicative identity) for safety
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Suppress numpy warnings for divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
    return np.where(np.abs(b) < 1e-10, 1.0, result)

def protected_sqrt(x):
    """Square root with negative input protection.

    Computes sqrt(|x|), returning 0.0 for negative values to avoid NaN.

    Args:
        x (array-like): Input value(s).

    Returns:
        np.ndarray: sqrt(|x|) where x >= 0, 0.0 where x < 0.

    Example:
        >>> protected_sqrt([4, -9, 0, 16])
        array([2.0, 0.0, 0.0, 4.0])
    """
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 0.0, np.sqrt(np.abs(x)))

def protected_log(x):
    """Natural logarithm with domain protection.

    Computes log(|x|), clamping to minimum 1e-10 to avoid log(0) and
    returning 0.0 for NaN/Inf results.

    Args:
        x (array-like): Input value(s).

    Returns:
        np.ndarray: log(|x|) where valid, 0.0 for invalid inputs.

    Example:
        >>> protected_log([np.e, 1, 0, -5])
        array([1.0, 0.0, 0.0, 1.609...])

    Notes:
        - Uses absolute value to handle negative inputs
        - Clamps to [1e-10, ∞) before taking logarithm
    """
    x = np.asarray(x, dtype=float)
    safe = np.clip(np.abs(x), 1e-10, None)  # avoid log(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(safe)
    return np.where(np.isnan(result) | np.isinf(result), 0.0, result)

def protected_pow(x, y):
    """Power function with exponent clamping and protection.

    Computes |x|^y with y clamped to [-3, 3] to prevent overflow/underflow.
    Returns 1.0 for NaN/Inf results.

    Args:
        x (array-like): Base value(s).
        y (array-like): Exponent(s).

    Returns:
        np.ndarray: |x|^(clamped y) where valid, 1.0 for invalid results.

    Example:
        >>> protected_pow([2, 3, 0], [3, 100, -1])
        array([8.0, 27.0, 1.0])

    Notes:
        - Uses absolute value of base to avoid complex numbers
        - Clamps exponents to prevent numerical explosions
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_clamped = np.clip(y, -3, 3)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.power(np.abs(x), y_clamped)
    return np.where(np.isnan(result) | np.isinf(result), 1.0, result)

def protected_exp(x):
    """Exponential function with overflow protection.

    Computes exp(x) with x clamped to [-50, 50] to prevent overflow.
    Returns 1e100 for infinite results.

    Args:
        x (array-like): Input value(s).

    Returns:
        np.ndarray: exp(clamped x), max 1e100.

    Example:
        >>> protected_exp([0, 1, 100, -50])
        array([1.0, 2.718..., 1e100, 1.928...e-22])

    Notes:
        - Clamping prevents overflow to infinity
        - Limit 1e100 provides large but finite result
    """
    x = np.asarray(x, dtype=float)
    x_clamped = np.clip(x, -50, 50)  # Prevent overflow
    with np.errstate(over='ignore'):
        result = np.exp(x_clamped)
    return np.where(np.isinf(result), 1e100, result)

def protected_reciprocal(x):
    """Reciprocal (1/x) with zero-divisor protection.

    Computes 1/x element-wise, returning 1.0 where |x| < 1e-10.

    Args:
        x (array-like): Input value(s).

    Returns:
        np.ndarray: 1/x where x is non-zero, 1.0 otherwise.

    Example:
        >>> protected_reciprocal([2, 0.5, 0, -4])
        array([0.5, 2.0, 1.0, -0.25])

    Notes:
        - Similar to protected_div(1, x) but specialized
        - Threshold 1e-10 prevents near-zero instability
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(1.0, x, out=np.ones_like(x, dtype=float), where=np.abs(x) >= 1e-10)
    return np.where(np.isnan(result) | np.isinf(result), 1.0, result)

def protected_not(x):
    """Boolean negation (element-wise NOT).

    Computes logical NOT for boolean array(s).

    Args:
        x (array-like): Boolean value(s).

    Returns:
        np.ndarray[bool]: Negated boolean array.

    Example:
        >>> protected_not([True, False, True])
        array([False, True, False])
    """
    x = np.asarray(x, dtype=bool)
    return ~x
