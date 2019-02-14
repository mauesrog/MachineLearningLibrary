"""General Utilities.

Defines miscellaneous methods.

"""
def appendargs(fn, *c_args):
    """Argument Function Right Binder.

    Wraps the given function with a handler that enforces the provided arguments
    to always be added to the right.

    Args:
        fn (callable): Function to wrap.
        *c_args (tuple): Arguments to always add to the right of all function
            calls involving `fn`.

    Returns:
        callable: Wrapped function or `None` if `fn` is invalid.

    """
    if not fn:
        return None

    if len(c_args) == 0:
        raise AttributeError("No arguments provided to append.")

    def wrapper(*args, **kwargs):
        """Function Wrapper.

        Adds the given arguments to the right of all function calls involving
        `fn`.

        Args:
            *args: Arbitrary arguments.
            **kwargs: Arbitrary keyword arguments.

        """
        return fn(*(args + c_args), **kwargs)

    return wrapper


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

    def composed_fn(*args, **kwargs):
        """Composed Function.

        Will perform the function composition accordin to the example (see
        example in `compose`).

        Args:
            *args: Arbitrary arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Value returned by all composed functions.

        """
        fn_copies = list(fns)

        fn_copies.reverse()

        return _compose_helper(fn_copies, *args, **kwargs)

    return composed_fn

def prependargs(fn, *c_args):
    """Argument Function Left Binder.

    Wraps the given function with a handler that enforces the provided arguments
    to always be added to the left.

    Args:
        fn (callable): Function to wrap.
        *c_args (tuple): Arguments to always add to the left of all function
            calls involving `fn`.

    Returns:
        callable: Wrapped function or `None` if `fn` is invalid.

    """
    if not fn:
        return None

    if len(c_args) == 0:
        raise AttributeError("No arguments provided to `prepend.")

    def wrapper(*args, **kwargs):
        """Function Wrapper.

        Adds the given arguments to the left of all function calls involving
        `fn`.

        Args:
            *args: Arbitrary arguments.
            **kwargs: Arbitrary keyword arguments.

        """
        return fn(*(c_args + args), **kwargs)

    return wrapper

def _compose_helper(fns, *args, **kwargs):
    """Function Composer Helper.

    Recursively calls the given functions in order with the arguments provided
    (which will get consumed by the very first function).

    Args:
        fns (tuple of callable): Functions to compose, provided in the exact
            order they should get consumed.
        *args: Arguments to feed the very first function i.e. `fns[0]`.
        **kwargs: Keyword arguments to feed the very first function i.e.
            `fns[0]`.

    Returns:
        The result of composing all functions with the given arguments, `None`
        if no functions were provided.

    Raises:
        TypeError: If `fns` contains a non-callable element.

    """
    fn = fns.pop()
    """callable: Next function in composition stack."""

    if len(fns) == 0:
        return fn(*args, **kwargs)

    return fn(_compose_helper(fns, *args, **kwargs))
