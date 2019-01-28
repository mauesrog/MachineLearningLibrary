"""General Utilities.

Defines miscellaneous methods.

"""
def compose(*fns):
    """Function Composer.

    Creates a callable composition with the functions provided. Useful for
    readability and decluttering.

    Example:
        `compose(f, e, d, c, b, a)(*args)` is equivalent to
        `f(e(d(c(b(a(*args))))))`.

    Args:
        *fns (tuple of callable): Functions to compose.

    Returns:
        callable: The function composition capable of receiving *args, `None`
            if no functions were provided.

    """
    # Needs at least one function to compose
    if len(fns) == 0:
        return None

    return lambda *args: _compose_helper(list(fns), args)

def _compose_helper(fns, args):
    """Function Composer Helper.

    Calls the given functions in order with the arguments provided (which will
    get consumed by the very first function).

    Args:
        fns (tuple of callable): Functions to compose, provided in the exact
            order they should get consumed.
        args: Arguments to feed the very first function i.e. `fns[0]`.

    Returns:
        The result of composing all functions with the given arguments, `None`
        if no functions were provided.

    Raises:
        TypeError: If `fns` contains a non-callable element.

    """
    result = fns.pop()(*args)
    """Keeps track of the running result of the function composition, started
    by feeding the given arguments to the first functions."""

    # Iterate until all functions have been consumed.
    while len(fns):
        result = fns.pop()(result)

    return result
