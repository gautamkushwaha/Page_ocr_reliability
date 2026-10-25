
#!/usr/bin/env python3
"""
Memory-aware, one-image feature-extraction pilot for Qwen3-VL-8B-Instruct.

Stages:
1. Select one image from the existing test manifest.
2. Generate an OCR transcription.
3. Compute token-level generation uncertainty.
4. Capture all vision-block outputs on CPU.
5. Capture DeepStack features through a vision-model forward hook.
6. Run a separate forward pass to inspect decoder hidden-state shapes.
7. Save outputs and stage-specific diagnostics.

This is a diagnostic pilot, not the full paired-page pipeline.

Run from:
    ~/GP/Hindi_OCR/page_ocr_reliability

    python src/pilot_extract.py 2>&1 | tee outputs/logs/pilot_extract.log
"""

import gc
import json
import logging
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

MANIFEST = Path("outputs/manifests/test_manifest.csv")
OUT_DIR = Path("outputs/pilot/one_page")

MAX_NEW_TOKENS = 1024
DEVICE_MAP = "auto"
ATTENTION_IMPLEMENTATION = "sdpa"


PROMPT = (
    "You are an accurate OCR system for examination sheets containing "
    "printed questions and handwritten answers. Extract the text and "
    "organize it into question-answer format.\n\n"

    "Requirements:\n"
    "- Printed text, including question numbers and answer choices, "
    "belongs to the question.\n"
    "- Handwritten text belongs to the student's answer.\n"
    "- Write each printed question first, followed by its handwritten "
    "answer on the next line.\n"
    "- IMPORTANT: If a handwritten word fills a blank or appears inside "
    "printed brackets, do NOT include that word in the question line. "
    "Keep the printed blank and printed answer choices as they appear, "
    "and write the handwritten word on the next line as the answer.\n"
    "- For example, if the printed question is "
    "'सीता खाना__है (खाता/खाती)' and the handwritten answer is 'खाती', "
    "output:\n"
    "4. सीता खाना__है (खाता/खाती)\n"
    "खाती\n"
    "- Preserve the original question order, wording, spelling, numbers, "
    "and symbols. Do not correct spelling or grammar.\n"
    "- Associate each answer with its question using the question number "
    "and page layout. Do not guess or invent missing text or answers.\n"
    "- For two-column layouts, read the entire left column from top to "
    "bottom, then the right column.\n"
    "- Write mathematics in plain text, using a^2 for powers and a/b "
    "for fractions. Do not use LaTeX commands or math delimiters.\n"
    "- Do not solve questions, complete answers, summarize, or translate.\n"
    "- Output only the extracted text, without explanations.\n\n"

    "Output format:\n"
    "1. [Printed question]\n"
    "[Handwritten answer]\n"
    "2. [Printed question]\n"
    "[Handwritten answer]"
)


LOG = logging.getLogger("pilot_extract")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def tensor_info(tensor):
    if not isinstance(tensor, torch.Tensor):
        return {"python_type": type(tensor).__name__}

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }


def cleanup_cuda():
    """Release unused cached GPU memory, not memory held by live tensors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_memory_info():
    """Return current CUDA memory statistics for diagnostic logging."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    result = {"cuda_available": True}

    for index in range(torch.cuda.device_count()):
        result[str(index)] = {
            "device_name": torch.cuda.get_device_name(index),
            "allocated_gib": round(
                torch.cuda.memory_allocated(index) / (1024 ** 3), 3
            ),
            "reserved_gib": round(
                torch.cuda.memory_reserved(index) / (1024 ** 3), 3
            ),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated(index) / (1024 ** 3), 3
            ),
        }

    return result


def move_inputs_to_model_device(inputs, model):
    """
    Move processor tensors to the input embedding device.

    With device_map='auto', Transformers dispatch hooks manage model
    submodules. This helper places input tensors at the model entry point.
    """
    device = model.get_input_embeddings().weight.device

    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def select_test_image(manifest_path: Path):
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path.resolve()}"
        )

    df = pd.read_csv(manifest_path)

    required = {"document_id", "page_a"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    for _, row in df.iterrows():
        raw_path = row["page_a"]

        if pd.isna(raw_path) or not str(raw_path).strip():
            continue

        path = Path(str(raw_path)).expanduser()

        if path.is_file():
            return str(row["document_id"]), path.resolve()

    raise FileNotFoundError(
        f"No existing page A image was found in {manifest_path}"
    )


def compute_token_uncertainty(scores):
    """
    Compute uncertainty one generation step at a time.

    scores is the tuple returned by generate(output_scores=True).
    Each element typically has shape [batch_size, vocabulary_size].

    Only compact per-token statistics are retained in the output.
    The original generation result must subsequently be deleted to
    release its score tensors.
    """
    token_probability = []
    entropy = []
    top1_top2_margin = []

    for step_index, step_scores in enumerate(scores):
        # Batch size is one in this pilot.
        logits = step_scores[0].detach().float()

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        token_entropy = -(probs * log_probs).sum()
        top2 = torch.topk(probs, k=2).values

        token_probability.append(float(top2[0].item()))
        entropy.append(float(token_entropy.item()))
        top1_top2_margin.append(float((top2[0] - top2[1]).item()))

        del logits, log_probs, probs, token_entropy, top2

        if (step_index + 1) % 64 == 0:
            LOG.info(
                "Computed uncertainty for %d generation steps",
                step_index + 1,
            )

    return {
        "token_probability": token_probability,
        "entropy": entropy,
        "top1_top2_margin": top1_top2_margin,
    }


def save_tensor_cpu(tensor):
    """Copy a feature tensor to CPU fp16 for diagnostic storage."""
    if not isinstance(tensor, torch.Tensor):
        return None

    return tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()


def extract_tensors_from_output(output):
    """
    Inspect common model-output containers without assuming one exact
    Transformers output class.

    Returns named tensor references for diagnostics. This does not
    recursively inspect arbitrary Python objects.
    """
    found = {}

    if isinstance(output, torch.Tensor):
        found["tensor"] = output
        return found

    if isinstance(output, (tuple, list)):
        for i, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                found[str(i)] = item
        return found

    # ModelOutput objects usually expose keys() and attribute access.
    if hasattr(output, "keys"):
        try:
            for key in output.keys():
                value = getattr(output, key, None)
                if isinstance(value, torch.Tensor):
                    found[str(key)] = value
                elif isinstance(value, (tuple, list)):
                    for i, item in enumerate(value):
                        if isinstance(item, torch.Tensor):
                            found[f"{key}_{i}"] = item
        except Exception:
            pass

    return found


# ---------------------------------------------------------------------
# Main pilot
# ---------------------------------------------------------------------

def main():
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_started = time.time()

    metadata = {
        "checkpoint": MODEL_ID,
        "manifest": str(MANIFEST),
        "output_directory": str(OUT_DIR),
        "max_new_tokens": MAX_NEW_TOKENS,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "stages": {},
    }

    document_id, image_path = select_test_image(MANIFEST)

    metadata["document_id"] = document_id
    metadata["image_path"] = str(image_path)

    LOG.info("Document ID: %s", document_id)
    LOG.info("Image path: %s", image_path)

    # Validate and load the image explicitly.
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    metadata["image"] = {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": "RGB",
    }

    # -------------------------------------------------------------
    # Load processor and model
    # -------------------------------------------------------------

    LOG.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    LOG.info("Loading model...")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    metadata["model_dtype"] = str(dtype)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=DEVICE_MAP,
        low_cpu_mem_usage=True,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    model.eval()

    metadata["model_class"] = type(model).__name__
    metadata["initial_gpu_memory"] = gpu_memory_info()

    # -------------------------------------------------------------
    # Register hooks for vision blocks and the full vision output
    # -------------------------------------------------------------

    vision_outputs = {}
    vision_forward_summary = {}
    hook_errors = []
    hooks = []

    visual = model.model.visual

    def make_block_hook(layer_index):
        def hook(module, hook_inputs, output):
            try:
                value = output[0] if isinstance(output, tuple) else output

                if isinstance(value, torch.Tensor):
                    # Immediately copy off GPU; do not retain activations
                    # on GPU through the generation process.
                    vision_outputs[layer_index] = save_tensor_cpu(value)
            except Exception as exc:
                hook_errors.append(
                    f"vision_block_{layer_index}: "
                    f"{type(exc).__name__}: {exc}"
                )

        return hook

    for layer_index, block in enumerate(visual.blocks):
        hooks.append(
            block.register_forward_hook(make_block_hook(layer_index))
        )

    def vision_model_hook(module, hook_inputs, output):
        """
        Capture the vision model's actual returned tensors, including
        DeepStack tensors if exposed by this installed Transformers build.
        """
        try:
            tensors = extract_tensors_from_output(output)

            summary = {
                "output_class": type(output).__name__,
                "tensor_outputs": {},
            }

            for name, tensor in tensors.items():
                summary["tensor_outputs"][name] = tensor_info(tensor)

                # Preserve DeepStack outputs and final vision output on CPU.
                # Names depend on the installed Transformers output class.
                is_deepstack = "deepstack" in name.lower()
                is_final = name in {
                    "last_hidden_state",
                    "tensor",
                    "0",
                }

                if is_deepstack or is_final:
                    vision_forward_summary.setdefault("saved_tensors", {})[
                        name
                    ] = save_tensor_cpu(tensor)

            vision_forward_summary["summary"] = summary

        except Exception as exc:
            hook_errors.append(
                f"vision_model_output: {type(exc).__name__}: {exc}"
            )

    hooks.append(visual.register_forward_hook(vision_model_hook))

    # -------------------------------------------------------------
    # Prepare multimodal inputs
    # -------------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    LOG.info("Preparing multimodal input...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)
    inputs = move_inputs_to_model_device(inputs, model)

    metadata["input_tensors"] = {
        key: tensor_info(value)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    LOG.info(
        "Input tensor metadata:\n%s",
        json.dumps(metadata["input_tensors"], indent=2),
    )

    # -------------------------------------------------------------
    # Stage 1: OCR generation
    # -------------------------------------------------------------

    LOG.info("Stage 1: generating transcription...")
    generation_started = time.time()

    generation_error = None
    result = None
    transcription = ""
    token_stats = {}

    try:
        with torch.inference_mode():
            result = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = result.sequences[:, input_length:]
        generated_token_count = int(generated_ids.shape[1])

        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Check whether the final generated token is a configured EOS token.
        eos_token_ids = model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_ids = set()
        elif isinstance(eos_token_ids, int):
            eos_ids = {eos_token_ids}
        else:
            eos_ids = set(eos_token_ids)

        last_generated_id = (
            int(generated_ids[0, -1].item())
            if generated_token_count > 0
            else None
        )

        reached_token_limit = (
            generated_token_count >= MAX_NEW_TOKENS
            and last_generated_id not in eos_ids
        )

        # Compute compact uncertainty features. Do not stack logits.
        token_stats = compute_token_uncertainty(result.scores or ())

        metadata["stages"]["generation"] = {
            "status": "success",
            "duration_seconds": round(time.time() - generation_started, 3),
            "generated_token_count": generated_token_count,
            "transcription_characters": len(transcription),
            "reached_max_new_tokens": reached_token_limit,
            "last_generated_token_id": last_generated_id,
            "uncertainty_steps": len(token_stats["entropy"]),
        }

        # Save now, before releasing tensors.
        (OUT_DIR / "transcription.txt").write_text(
            transcription,
            encoding="utf-8",
        )

        save_json(OUT_DIR / "token_statistics.json", token_stats)

        # Store token IDs separately for reproducibility.
        save_json(
            OUT_DIR / "generated_token_ids.json",
            generated_ids[0].detach().cpu().tolist(),
        )

        del generated_ids

    except Exception as exc:
        generation_error = f"{type(exc).__name__}: {exc}"
        metadata["stages"]["generation"] = {
            "status": "failed",
            "error": generation_error,
            "traceback": traceback.format_exc(),
        }
        raise

    finally:
        # Remove hooks even if generation fails.
        for hook in hooks:
            hook.remove()

    # The scores in result can occupy substantial GPU memory.
    del result
    cleanup_cuda()

    metadata["generation_gpu_memory_after_cleanup"] = gpu_memory_info()

    # -------------------------------------------------------------
    # Stage 2: save captured vision representations
    # -------------------------------------------------------------

    LOG.info("Stage 2: saving vision block outputs...")

    vision_dir = OUT_DIR / "vision_blocks"
    vision_dir.mkdir(parents=True, exist_ok=True)

    vision_shapes = {}
    for layer_index, tensor in sorted(vision_outputs.items()):
        path = vision_dir / f"vision_block_{layer_index:02d}.pt"
        torch.save(tensor, path)

        vision_shapes[str(layer_index)] = {
            **tensor_info(tensor),
            "file": str(path),
        }

    metadata["stages"]["vision_blocks"] = {
        "status": "success" if vision_outputs else "no_outputs",
        "captured_block_count": len(vision_outputs),
        "blocks": vision_shapes,
    }

    # Save the final vision tensor and DeepStack tensors if returned by
    # the model's vision forward output.
    deepstack_dir = OUT_DIR / "vision_features"
    deepstack_dir.mkdir(parents=True, exist_ok=True)

    vision_feature_summary = vision_forward_summary.get(
        "summary",
        {"warning": "Vision model output hook did not capture metadata."},
    )

    saved_vision_tensors = vision_forward_summary.get("saved_tensors", {})
    saved_vision_feature_info = {}

    for name, tensor in saved_vision_tensors.items():
        safe_name = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in name
        )
        path = deepstack_dir / f"{safe_name}.pt"
        torch.save(tensor, path)

        saved_vision_feature_info[name] = {
            **tensor_info(tensor),
            "file": str(path),
        }

    metadata["stages"]["vision_forward_output"] = {
        "status": "success" if vision_feature_summary else "no_output",
        "summary": vision_feature_summary,
        "saved_tensors": saved_vision_feature_info,
    }

    metadata["hook_errors"] = hook_errors

    # Release vision tensors from Python references after saving.
    vision_outputs.clear()
    saved_vision_tensors.clear()
    vision_forward_summary.clear()
    cleanup_cuda()

    # -------------------------------------------------------------
    # Stage 3: inspect decoder hidden states
    # -------------------------------------------------------------

    LOG.info("Stage 3: inspecting decoder hidden states...")
    decoder_started = time.time()

    decoder_hidden_shapes = []
    decoder_error = None

    try:
        # Do not retain the resulting activations beyond shape inspection.
        with torch.inference_mode():
            forward_out = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = getattr(forward_out, "hidden_states", None)

        if hidden_states is not None:
            decoder_hidden_shapes = [
                tensor_info(tensor)
                for tensor in hidden_states
                if isinstance(tensor, torch.Tensor)
            ]

        output_keys = (
            list(forward_out.keys())
            if hasattr(forward_out, "keys")
            else []
        )

        metadata["stages"]["decoder_hidden_states"] = {
            "status": "success" if decoder_hidden_shapes else "no_hidden_states",
            "duration_seconds": round(time.time() - decoder_started, 3),
            "hidden_state_count": len(decoder_hidden_shapes),
            "hidden_state_shapes": decoder_hidden_shapes,
            "model_forward_output_keys": output_keys,
            "note": (
                "This pilot records decoder hidden-state shapes only; "
                "it does not save full decoder activation tensors."
            ),
        }

        del forward_out
        del hidden_states
        cleanup_cuda()

    except Exception as exc:
        decoder_error = f"{type(exc).__name__}: {exc}"
        metadata["stages"]["decoder_hidden_states"] = {
            "status": "failed",
            "error": decoder_error,
            "traceback": traceback.format_exc(),
        }
        LOG.exception("Decoder hidden-state inspection failed.")
        cleanup_cuda()

    finally:
        # Pixel tensors are no longer needed after vision and decoder passes.
        inputs.clear()
        cleanup_cuda()

    # -------------------------------------------------------------
    # Stage 4: summarize token uncertainty
    # -------------------------------------------------------------

    entropy_values = token_stats.get("entropy", [])
    probability_values = token_stats.get("token_probability", [])
    margin_values = token_stats.get("top1_top2_margin", [])

    def mean_or_none(values):
        return sum(values) / len(values) if values else None

    metadata["generation_uncertainty_summary"] = {
        "token_count": len(entropy_values),
        "mean_token_probability": mean_or_none(probability_values),
        "mean_entropy_nats": mean_or_none(entropy_values),
        "mean_top1_top2_margin": mean_or_none(margin_values),
        "note": (
            "Entropy is calculated from each generation-step vocabulary "
            "distribution. These are generation features, not GT-derived labels."
        ),
    }

    # -------------------------------------------------------------
    # Final metadata
    # -------------------------------------------------------------

    metadata["final_gpu_memory"] = gpu_memory_info()
    metadata["total_duration_seconds"] = round(time.time() - run_started, 3)

    save_json(OUT_DIR / "pilot_metadata.json", metadata)

    LOG.info("Pilot finished.")
    LOG.info("Transcription preview:\n%s", transcription[:1500])
    LOG.info("Vision blocks captured: %d", len(vision_shapes))
    LOG.info(
        "Decoder hidden-state count: %d",
        len(decoder_hidden_shapes),
    )
    LOG.info(
        "Vision output summary:\n%s",
        json.dumps(vision_feature_summary, indent=2),
    )
    LOG.info("Stage results:\n%s", json.dumps(metadata["stages"], indent=2))
    LOG.info("Saved outputs to: %s", OUT_DIR)


if __name__ == "__main__":
    main()
