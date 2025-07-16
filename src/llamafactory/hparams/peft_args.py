from dataclasses import dataclass
from peft import PeftConfig
from adapters import AdapterConfig

@dataclass
class PeftArguments(PeftConfig, AdapterConfig):
    pass