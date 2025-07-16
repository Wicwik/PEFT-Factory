from peft import PeftConfig
from dataclasses import dataclass, field

from typing import Optional

@dataclass
class AdaptersConfig():
    adapters : list[str] = None
    classification_head : Optional[bool] = False
