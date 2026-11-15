import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from utils.annotation.qwen_batch import class_id_constraint


class Digits:
    def encode(self, value, add_special_tokens=False):
        assert not add_special_tokens
        return [int(digit) for digit in value]
    def decode(self, tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(map(str, tokens))


def test_multitoken_class_prefixes_left_padding_and_finished_rows():
    allowed = class_id_constraint(Digits(), 101, 3, [98, 99], 4)
    prefix = [777, 0, 777]  # Whole padded prompt, including any padding.
    assert set(allowed(0, torch.tensor(prefix))) == set(range(10))
    assert set(allowed(0, torch.tensor(prefix + [1]))) == set(range(10)) | {98, 99}
    assert set(allowed(0, torch.tensor(prefix + [1, 0]))) == {0, 98, 99}
    assert set(allowed(0, torch.tensor(prefix + [1, 0, 0]))) == {98, 99}
    assert allowed(1, torch.tensor(prefix + [1, 98, 777, 777])) == [98, 99]


def test_47_class_constraint_prevents_out_of_range_48_and_empty_output():
    allowed = class_id_constraint(Digits(), 47, 2, 99, 3)
    assert 99 not in allowed(0, torch.tensor([777, 777]))
    assert 8 not in allowed(0, torch.tensor([777, 777, 4]))
    assert set(allowed(0, torch.tensor([777, 777, 4]))) == {0, 1, 2, 3, 4, 5, 6, 99}
    with pytest.raises(ValueError, match="escaped"):
        allowed(0, torch.tensor([777, 777, 4, 8]))


def test_generation_budget_accounts_for_complete_id_and_eos():
    with pytest.raises(ValueError, match="longest class ID"):
        class_id_constraint(Digits(), 101, 2, 99, 3)
