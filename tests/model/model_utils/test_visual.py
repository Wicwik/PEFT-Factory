# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from transformers import AutoConfig, AutoModelForVision2Seq

from llamafactory.hparams import FinetuningArguments, ModelArguments
from llamafactory.model.adapter import init_adapter


@pytest.mark.parametrize("freeze_vision_tower", (False, True))
@pytest.mark.parametrize("freeze_multi_modal_projector", (False, True))
@pytest.mark.parametrize("freeze_language_model", (False, True))
def test_visual_full(freeze_vision_tower: bool, freeze_multi_modal_projector: bool, freeze_language_model: bool):
    model_args = ModelArguments(model_name_or_path="Qwen/Qwen2-VL-2B-Instruct")
    finetuning_args = FinetuningArguments(
        finetuning_type="full",
        freeze_vision_tower=freeze_vision_tower,
        freeze_multi_modal_projector=freeze_multi_modal_projector,
        freeze_language_model=freeze_language_model,
    )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with torch.device("meta"):
        model = AutoModelForVision2Seq.from_config(config)

    model = init_adapter(config, model, model_args, finetuning_args, peft_args=None, is_trainable=True)
    for name, param in model.named_parameters():
        if any(key in name for key in ["visual.patch_embed", "visual.blocks"]):
            assert param.requires_grad != freeze_vision_tower
        elif "visual.merger" in name:
            assert param.requires_grad != freeze_multi_modal_projector
        else:
            assert param.requires_grad != freeze_language_model


@pytest.mark.parametrize("freeze_vision_tower", (False, True))
def test_visual_lora(freeze_vision_tower: bool):
    model_args = ModelArguments(model_name_or_path="Qwen/Qwen2-VL-2B-Instruct")
    finetuning_args = FinetuningArguments(finetuning_type="lora", freeze_vision_tower=freeze_vision_tower)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with torch.device("meta"):
        model = AutoModelForVision2Seq.from_config(config)

    model = init_adapter(config, model, model_args, finetuning_args, peft_args=None, is_trainable=True)
    trainable_params, frozen_params = set(), set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.add(name)
        else:
            frozen_params.add(name)

    if freeze_vision_tower:
        assert "base_model.model.visual.blocks.0.attn.qkv.lora_A.default.weight" not in trainable_params
    else:
        assert "base_model.model.visual.blocks.0.attn.qkv.lora_A.default.weight" in trainable_params

    assert "merger" not in trainable_params
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight" in trainable_params
