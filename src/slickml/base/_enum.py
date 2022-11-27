from enum import Enum
from typing import Dict, List


class ExtendedEnum(Enum):
    """Base Enum type with compatible string functionalities.

    Methods
    -------
    names()
        Returns a list of Enum names as string

    values()
        Returns a list of Enum values as string

    to_dict()
        Returns a dictionary of all Enum name-value pairs

    Examples
    --------
    >>> from slickml.utils import ExtendedEnum
    >>> class FooBar(ExtendedEnum):
    ...    FOO = "foo"
    ...    BAR = "bar"
    >>> FooBar.FOO
    >>> FooBar.names()
    >>> FooBar.values()
    >>> FooBar.to_dict()
    """

    @classmethod
    def names(cls) -> List[str]:
        """Returns a list of Enum names as string.

        Returns
        -------
        List[str]
        """
        return [c.name for c in cls]

    @classmethod
    def values(cls) -> List[str]:
        """Returns a list of Enum values as string.

        Returns
        -------
        List[str]
        """
        return [c.value for c in cls]

    @classmethod
    def to_dict(cls) -> Dict[str, str]:
        """Returns a dictionary of all Enum name-value pairs as string.

        Returns
        -------
        Dict[str, str]
        """
        return {name: str(value) for (name, value) in cls.__members__.items()}

    def __str__(self) -> str:
        """Returns the Enum ``str`` value.

        Returns
        -------
        str
        """
        return self.value

    def __repr__(self) -> str:
        """Returns the Enum ``str`` representation value.

        Returns
        -------
        str
        """
        return f"'{self.value}'"
