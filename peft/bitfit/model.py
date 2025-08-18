from peft import BaseTuner

class BitFitModel(BaseTuner):
    prefix: str = "ln_tuning_"

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":  
                raise
            return getattr(self.model, name)