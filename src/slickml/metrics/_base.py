from typing import Protocol


# TODO(amir): docstring and more functions ?
class Metric(Protocol):
    def plot(self) -> None:
        ...
