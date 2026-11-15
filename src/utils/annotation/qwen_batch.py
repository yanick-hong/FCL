"""Greedy batched Qwen2.5-VL prediction with explicit visual and model settings."""
from __future__ import annotations

import importlib.metadata


class BatchedQwenPredictor:
    def __init__(self, model_name, revision="main", min_pixels=200704, max_pixels=200704,
                 dtype="bfloat16", attention="sdpa"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision,
            min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,
            revision=revision, torch_dtype=getattr(torch, dtype), device_map="auto",
            attn_implementation=attention)
        self.model.eval()
        self.metadata = {"requested_model": model_name, "requested_revision": revision,
                         "resolved_commit": getattr(self.model.config, "_commit_hash", None),
                         "dtype": dtype, "attention": attention,
                         "min_pixels": min_pixels, "max_pixels": max_pixels,
                         "packages": {name: importlib.metadata.version(name) for name in
                                      ("torch", "transformers", "accelerate", "Pillow")}}

    def __call__(self, paths, prompt, max_new_tokens=16):
        from PIL import Image
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text] * len(images), images=images,
                                padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False,
                                             max_new_tokens=max_new_tokens, use_cache=True)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
