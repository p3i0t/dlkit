from abc import ABC

# from dataclasses import dataclass
#
# from optimus.exceptions import ArgumentNotProvidedError


# @dataclass(frozen=True, kw_only=True)
class ModelConfig(ABC):
    def to_dict(self):
        return vars(self)

    # name: str = None
    # d_in: int = None
    # d_out: int = None
    #
    # def __post_init__(self):
    #     if self.name is None:
    #         raise ArgumentNotProvidedError('name should be provided.')
    #     if self.d_in is None:
    #         raise ArgumentNotProvidedError('d_in should be provided.')
    #     if self.d_out is None:
    #         raise ArgumentNotProvidedError('d_out should be provided.')
