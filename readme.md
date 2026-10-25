# Paired-Page OCR Feature Extraction

## 1. What are we doing?

This project uses **Qwen3-VL-8B-Instruct** to extract text and internal model features from Hindi examination sheets containing printed questions and handwritten answers.

We process page A and page B separately, then combine their features for document-level analysis.

### Why are we doing this?

Our research goal is to investigate whether a model's internal features and generation uncertainty can help identify when OCR predictions are likely to contain errors.

We extract:
- **OCR text:** The text generated from each page.
- **Vision features:** Representations from selected vision encoder layers.
- **DeepStack features:** Intermediate visual features exposed by Qwen3-VL.
- **Decoder features:** Representations from selected language-model layers.
- **Token uncertainty:** Token probabilities, entropy, and probability margins.

These features will later be compared with ground-truth transcriptions to investigate their relationship with OCR errors.

**Note:** This code extracts features; it does not train an error-prediction model.

## 2. Files

- `Paired_Page_OCR_Extractor.py` — Main executable Python script.
- `pilot_extract.py` -- pilot to check 
- `readme.md` — Detailed explanation of the research methodology.

Only the Python script is required to run the pipeline.

## 3. Requirements

Install the required packages in your Python environment:

```bash
pip install numpy pandas pillow transformers
```

Install PyTorch separately according to your system and CUDA version: https://pytorch.org/get-started/locally/

The Qwen3-VL model must also be accessible through Hugging Face.

## 4. Input data

The script expects a manifest CSV containing these columns:

| Column | Meaning |
|---|---|
| `document_id` | Unique document identifier |
| `page_a` | Path to page A image |
| `page_b` | Path to page B image |

By default, the test manifest is:

```text
outputs/manifests/test_manifest.csv
```

The default dataset root is configured inside the Python script. Update it if your images are stored elsewhere.

## 5. How to run

Run from the project root.

First, test three documents:

```bash
python src/Unified_Paired_Page_OCR_Extractor.py \
  --split test \
  --limit 3 \
  --max-new-tokens 1024 \
  --output-root outputs/paired_pilot
```

Check the output and logs. If everything works, process all selected documents:

```bash
python src/Unified_Paired_Page_OCR_Extractor.py \
  --split test \
  --limit 0 \
  --max-new-tokens 1024 \
  --output-root outputs/paired_test
```

Use a different output directory for a new experiment or configuration.

## 6. Main outputs

The pipeline saves:

- `transcription.txt` — OCR text for each page.
- `token_statistics.json` — Token-level uncertainty statistics.
- `compact_features.pt` — Extracted feature vectors for each page.
- `page_metadata.json` — Feature dimensions, configuration, and execution status.
- `paired_features.pt` — Combined page A and page B features.
- `merged_predictions/` — Combined OCR transcriptions.
- `page_feature_index.csv` — Index of extracted page features.
- `page_status.json` — Processing status for each page.

## 7. Research workflow

1. Extract OCR text and features.
2. Compare OCR predictions against ground-truth transcriptions.
3. Calculate metrics such as Character Error Rate (CER) and Word Error Rate (WER).
4. Study how the extracted features relate to OCR errors.
5. Train and evaluate a separate model to predict OCR reliability.

Keep training, validation, and test data separate to avoid data leakage.

## 8. Important

Start with a small pilot before processing the full dataset. Verify the transcriptions, feature files, and processing status.

Token uncertainty and internal representations are only candidate reliability signals. Their usefulness must be established through experiments.

