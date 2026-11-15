"""Greedy batched Qwen2.5-VL prediction with explicit visual and model settings."""
from __future__ import annotations

import importlib.metadata


def class_id_constraint(tokenizer, num_classes, prompt_length, eos_token_ids, max_new_tokens=None):
    """A token trie of complete allowed IDs; no parsing repair or truth access."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("A positive class count is required for constrained decoding")
    end_ids = [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids)
    if not end_ids or any(type(value) is not int for value in end_ids):
        raise ValueError("Constrained decoding requires explicit EOS token IDs")
    trie, longest = {}, 0
    for label in range(num_classes):
        tokens = tokenizer.encode(str(label), add_special_tokens=False)
        longest = max(longest, len(tokens))
        if not tokens or any(token in set(end_ids) | set(getattr(tokenizer, "all_special_ids", [])) for token in tokens):
            raise ValueError("Class IDs must tokenize into nonempty nonspecial token sequences")
        if tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False) != str(label):
            raise ValueError("Class ID tokenization is not exactly reversible")
        node = trie
        for token in tokens:
            node = node.setdefault(token, {})
        node["terminal"] = True
    if max_new_tokens is not None and max_new_tokens < longest + 1:
        raise ValueError("Generation budget must allow the longest class ID plus EOS")

    def allowed(batch_id, input_ids):
        suffix = input_ids[prompt_length:].tolist()
        node = trie
        for index, token in enumerate(suffix):
            if token in end_ids and node.get("terminal"):
                # Transformers can query an already finished row while other
                # members of the batch are still generating.
                return end_ids
            if token not in node:
                raise ValueError(f"Generated prefix escaped the class ID trie at token {index}")
            node = node[token]
        result = [token for token in node if isinstance(token, int)]
        if node.get("terminal"):
            result.extend(end_ids)
        return result

    return allowed


class BatchedQwenPredictor:
    def __init__(self, model_name, revision="main", min_pixels=200704, max_pixels=200704,
                 dtype="bfloat16", attention="sdpa"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision,
            min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,
            revision=revision, torch_dtype=getattr(torch, dtype), device_map="auto",
            attn_implementation=attention)
        self.model.eval()
        self._constraints = {}
        self.metadata = {"requested_model": model_name, "requested_revision": revision,
                         "resolved_commit": getattr(self.model.config, "_commit_hash", None),
                         "dtype": dtype, "attention": attention,
                         "min_pixels": min_pixels, "max_pixels": max_pixels,
                         "processor_use_fast": True,
                         "image_processor_class": type(self.processor.image_processor).__name__,
                         "image_processor_config": self.processor.image_processor.to_dict(),
                         "device_map": {key: str(value) for key, value in self.model.hf_device_map.items()},
                         "generation_defaults": self.model.generation_config.to_dict(),
                         "class_constraint": {"algorithm": "class_id_token_trie", "version": 1,
                             "whitespace": "none", "eos_token_ids": self.model.generation_config.eos_token_id,
                             "pad_token_id": self.processor.tokenizer.pad_token_id},
                         "packages": {name: importlib.metadata.version(name) for name in
                                      ("torch", "transformers", "accelerate", "Pillow")}}

    def __call__(self, paths, prompt, max_new_tokens=16, classes=None):
        if classes is None:
            raise ValueError("Explicit ordered classes are required for controlled Qwen decoding")
        from PIL import Image
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text] * len(images), images=images,
                                padding=True, return_tensors="pt").to(self.model.device)
        end_ids = self.model.generation_config.eos_token_id or self.processor.tokenizer.eos_token_id
        key = (len(classes), inputs.input_ids.shape[1], max_new_tokens)
        if key not in self._constraints:
            self._constraints[key] = class_id_constraint(self.processor.tokenizer, len(classes),
                inputs.input_ids.shape[1], end_ids, max_new_tokens)
        constraints = {"prefix_allowed_tokens_fn": self._constraints[key]}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False,
                num_beams=1, num_return_sequences=1, min_length=0, min_new_tokens=0,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
                max_new_tokens=max_new_tokens, use_cache=True, **constraints)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
