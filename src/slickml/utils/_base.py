from typing import Any, Optional, Tuple, Union


# TODO(amir): figure out a way to extend validations (i.e. when we are validating list[values])
# currently, the values cannot be done for iterable i.e. tuple, list, and dict
# the easy way to pull it off is a for loop
def check_var(
    var: Any,
    *,
    var_name: str,
    dtypes: Union[type, Tuple[type]],
    values: Optional[Union[Any, Tuple[Any]]] = None,
) -> None:
    """Validates the variable's dtype and possible value.

    Parameters
    ----------
    var : Any
        Variable

    var_name : str
        Variable name

    dtypes : Union[type, Tuple[type]]
        Data type classes

    values : Optional[Union[Any, Tuple[Any]]], optional
        Possible values, by default None

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If dtypes are invalid

    ValueError
        If values are invalid
    """

    def _check_dtypes(
        var: Any,
        var_name: str,
        dtypes: Union[type, Tuple[type]],
    ) -> None:
        """Validates the variable's dtype.

        Parameters
        ----------
        var : Any
            Variable

        var_name : str
            Variable name

        dtypes : Union[type, Tuple[type]]
            Data type classes

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If dtypes are invalid
        """
        if not isinstance(dtypes, tuple):
            dtypes = (dtypes,)
            _sum = sum([isinstance(d, type) for d in dtypes])
            if _sum != len(dtypes):
                raise TypeError("The input dtypes must have one of [type, Tuple(type)] dtypes.")
        dtype_names = [s.__name__ for s in dtypes]
        if len(dtype_names) == 1:
            type_error_msg = f"The input {var_name} must have {dtype_names[0]} dtype."
        else:
            type_error_msg = f"The input {var_name} must have one of {dtype_names} dtypes."
        if not isinstance(var, dtypes):
            raise TypeError(type_error_msg)

    def _check_values(
        var: Any,
        var_name: str,
        values: Union[Any, Tuple[Any]],
    ) -> None:
        """Validates variable's possible value.

        Parameters
        ----------
        var : Any
            Variable

        var_name : str
            Variable name

        values : Union[Any, Tuple[Any]]
            Possible values

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If values are invalid
        """
        if not isinstance(values, tuple):
            values = (values,)
        if len(values) == 1:
            value_error_msg = f"The input {var_name} must have a value of {values[0]}."
        else:
            value_error_msg = f"The input {var_name} must have one of {values} values."
        if var not in values:
            raise ValueError(value_error_msg)

    _check_dtypes(
        var=var,
        var_name=var_name,
        dtypes=dtypes,
    )
    if values is not None:
        _check_values(
            var=var,
            var_name=var_name,
            values=values,
        )

    return None
