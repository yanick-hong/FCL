"""Verify the scoped manuscript revision and key requested edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "FCL_CVPR2027"
ORIGINAL = ROOT / "tmp" / "paper_revision" / "original"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    supp_tex = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    assert "Controlled selection and objective ablations" in supp_tex
    assert "Stability across Training Seeds" in supp_tex
    assert "Filter-Corrected Qwen Labels" in supp_tex
    assert "Random-30" in main_tex and "Random-M" in main_tex
    assert "FCL-C" in main_tex and "FCL-Q" in supp_tex
    assert ("102-class" in main_tex or "102 classes" in main_tex)
    assert ("102-class" in supp_tex or "102 classes" in supp_tex)
    assert "dagger" not in main_tex.lower() and "dagger" not in supp_tex.lower()
    # The requested query/count columns are absent from the two affected tables.
    table1 = main_tex[main_tex.index("\\label{tab:fcls-annotation}") : main_tex.index("\\end{table*}", main_tex.index("\\label{tab:fcls-annotation}"))]
    qwen = supp_tex[supp_tex.index("\\label{tab:supp-qwen-labels}") : supp_tex.index("\\end{table*}", supp_tex.index("\\label{tab:supp-qwen-labels}"))]
    assert "count" not in table1.lower() and "queries count" not in qwen.lower()
    assert "Train Samples & Classes & Qwen & UL-ER & FL-ER & Random-30 & Random-M & FCL-Q & Gain" in qwen
    assert "Caltech-101 & 2,945 & 102" in qwen
    assert "tab:component-controls" not in main_tex
    assert "tab:supp-controlled-selection-objective" not in supp_tex
    additional = supp_tex[supp_tex.index("\\label{tab:supp-additional-baselines}") : supp_tex.index("\\end{table*}", supp_tex.index("\\label{tab:supp-additional-baselines}"))]
    assert "Prototype + AUC-C" in additional
    assert "Random + AUC-C" in additional
    assert "Prototype + CLIP soft-CE" in additional
    method_table = main_tex[main_tex.index("\\label{tab:method-compare-clip}") : main_tex.index("\\end{table*}", main_tex.index("\\label{tab:method-compare-clip}"))]
    assert "Random-Filter-CE" in method_table and "FCL-Filter-CE" in method_table
    seed_table_start = supp_tex.index("\\label{tab:supp-controlled-training-seeds}")
    seed_table = supp_tex[supp_tex.rfind("\\begin{table}[t]", 0, seed_table_start) : supp_tex.index("\\end{table}", seed_table_start)]
    assert "Mean $\\pm$ SD" in seed_table and "95\\% CI" in seed_table
    assert "Seed 42" not in seed_table and "Seed 43" not in seed_table and "Seed 44" not in seed_table

    unchanged = []
    changed = []
    for original in ORIGINAL.rglob("*"):
        if not original.is_file():
            continue
        rel = original.relative_to(ORIGINAL)
        current = PAPER / rel
        if not current.exists():
            raise AssertionError(f"missing current file: {rel}")
        (changed if sha256(original) != sha256(current) else unchanged).append(str(rel).replace("\\", "/"))
    assert set(changed) == {"main.tex", "supplement.tex"}, changed

    result = {
        "status": "passed",
        "changed_files": changed,
        "unchanged_original_files": len(unchanged),
        "checks": {
            "controlled_ablation_block_in_supp_table1": True,
            "standalone_supp_table3_removed": True,
            "qwen_section": True,
            "table1_unified": True,
            "query_count_columns_removed": True,
            "caltech_102_class": True,
            "dagger_footnotes_absent": True,
            "component_controls_merged": True,
            "qwen_table_ten_columns": True,
            "seed_columns_removed": True,
        },
    }
    out = ROOT / "docs" / "results" / "paper_revision_verification_v3.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
