from dataclasses import dataclass, field

from peft import PeftConfig

@dataclass
class BitFitConfig(PeftConfig):
    layers : list[str] = field(default=["all"], metadata={"help": "Layer names that will be fine-tuned with BitFit. Default is all layers."})