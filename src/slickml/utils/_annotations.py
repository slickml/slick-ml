import functools
import warnings
from typing import Callable, Optional, TypeVar

# TODO(amir): check when `ParamSpec` will be available in `typing`? I guess it is available for
# python version >= 3.10 --> https://www.youtube.com/watch?v=fwZoxWyMGM8
# we might wanna import it in a `try/except block` depending on the python version to be bullet-proof
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def deprecated(
    alternative: Optional[str] = None,
    since: Optional[str] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Annotation decorator for marking APIs as deprecated in docstrings and raising a warning if called.

    Parameters
    ----------
    alternative : str, optional
        The name of a superseded replacement function, method, or class to use in place of the
        deprecated one, by default None

    since : str, optional
        A version designator defining during which release the function, method, or class was marked
        as deprecated, by default None

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
    """

    def deprecated_decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Main annotation decorator.

        Parameters
        ----------
        func : Callable[P, R]
            Deprecated function

        Returns
        -------
        Callable[P, R]
        """
        since_str = f" since {since}" if since else ""
        warning_message = f"""``{func.__module__}.{func.__qualname__}`` is deprecated{since_str} and will be removed in a future release."""
        if alternative is not None and alternative.strip():
            warning_message += f" Use ``{alternative}`` instead."

        @functools.wraps(func)
        def inner_deprecated_decorator(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            """Inner annotation decorator.

            Parameters
            ----------
            *args : P.args
                Positional arguments of the deprecated function

            **kwargs : P.kwargs
                Named arguments of the deprecated function

            Returns
            -------
            R
            """
            warnings.warn(
                warning_message,
                category=FutureWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        # TODO(amir): this is currently not working for classes properly
        if func.__doc__ is not None:
            inner_deprecated_decorator.__doc__ = (
                f".. deprecated:: {warning_message}\n{func.__doc__}"
            )
        else:
            inner_deprecated_decorator.__doc__ = f".. deprecated:: {warning_message}\n"
        return inner_deprecated_decorator

    return deprecated_decorator
