from dataclasses import dataclass

from adapters import DoubleSeqBnConfig


@dataclass
class AdaptersConfig:
    adapter_name: str = "default"

@dataclass
class AdaptersDoubleSeqBnConfig(AdaptersConfig, DoubleSeqBnConfig):
    # this is mostly because DoubleSeqBnConfig from adapters contains Union[X, Y] where both are not optional, so HFArgumentParser cannot parse it properly
    reduction_factor: float = 16
    residual_before_ln: bool = True
