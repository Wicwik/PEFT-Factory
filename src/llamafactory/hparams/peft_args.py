from dataclasses import dataclass

from adapters import AdapterConfig
from peft import PeftConfig


@dataclass
class PeftArguments(PeftConfig, AdapterConfig):
    pass
