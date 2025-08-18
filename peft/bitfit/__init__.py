from peft.utils import register_peft_method

from .config import BitFitConfig
from .model import BitFitModel

register_peft_method(name="bitfit", config_cls=BitFitConfig, model_cls=BitFitModel, is_mixed_compatible=False)