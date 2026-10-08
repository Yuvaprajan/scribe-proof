# ScribeProof evaluation datasets

## Layout (do not silently mix)

```text
evaluation/dataset/
  dev/              # synthetic / debug only — NEVER qualification evidence
  held_out/         # frozen preliminary set (may still contain synthetics)
  qualification/    # real hard pages only (is_synthetic=false); gate enforced
```

## Case schema (required for qualification)

```json
{
  "id": "hw_001",
  "image": "images/hw_001.png",
  "ground_truth": "line 1\nline 2",
  "source": "scanner_batch_2026",
  "is_synthetic": false,
  "difficulty": "extreme",
  "script": "latin_cursive",
  "tags": ["extreme", "medical", "cursive"]
}
```

Aliases accepted by loaders: `image_path`↔`image`, `difficulty_tags`↔`tags`.

## Qualification gate

- Minimum **15** real hard pages with human GT (`is_synthetic` MUST be false)
- Target **30** real hard pages
- Refuse `QUALIFIED` / challenge pass when: synthetics present, missing GT,
  duplicate images, cross-split leaks, placeholders, or OCR-generated GT

## Commands

From `backend/` (API running for ScribeProof rows):

```bash
python -m evaluation.dataset_integrity --root ../evaluation/dataset
python -m evaluation.run_baseline --dataset ../evaluation/dataset/held_out --independent --api http://localhost:8000
python -m evaluation.run_ablation --dataset ../evaluation/dataset/held_out --api http://localhost:8000
python -m evaluation.qualification_report --dataset-root ../evaluation/dataset
```

Current held_out n≈5 with GT (mostly synthetic) is **PRELIMINARY only**.
