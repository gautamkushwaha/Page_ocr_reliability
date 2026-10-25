#!/usr/bin/env python3
"""Resumable paired-page OCR + compact vision/DeepStack/decoder feature extraction in one generation call.
# to run full keep limit 0 and also  -- split test is the input directory. the base dir(DEFAULT_DATA_ROOT = Path('/DATA/gyan/GP/Hindi_OCR/data/RAIT/HEAD DATASET'))
for this code.
  python src/paired_page_extract.py --> path of your code
  --split test \ --> input of your image file
  --limit 3 --> how many pairs you want to run. IN this case 3 paris (A and B together)
  -- output-root outputs/paired_pilot --> output file path
  -- 2>&1 | tee outputs/logs/paired_page_extract.log  --> logs file path 
python src/paired_page_extract.py \
    --split test \
    --limit 3 \ 
    --max-new-tokens 1024 \
    --output-root outputs/paired_pilot \
    2>&1 | tee outputs/logs/paired_page_extract.log
"""
import argparse, gc, json, logging, re, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# note these are from pilot_extract as they are defined there as well. You can use in a single project as well
from pilot_extract import (MODEL_ID, PROMPT, MAX_NEW_TOKENS, DEVICE_MAP,
    ATTENTION_IMPLEMENTATION, compute_token_uncertainty,
    move_inputs_to_model_device, cleanup_cuda, gpu_memory_info, tensor_info)


import logging

# Mute HTTP request loggers from urllib3 and httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

DEFAULT_DATA_ROOT = Path('/DATA/gyan/GP/Hindi_OCR/data/RAIT/HEAD DATASET')
# Vision encoder layers to pool. Keep the established pilot selection.
SELECTED_BLOCKS = (0, 8, 16, 26)

# Four decoder layers, selected by relative depth at runtime so this works
# with the actual language-model depth in the installed Qwen3-VL checkpoint.
SELECTED_DECODER_LAYER_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
EXTRACTION_SCHEMA_VERSION = 4

EXPECTED_FEATURE_KEYS = tuple(
    [f'vision_block_{i:02d}' for i in SELECTED_BLOCKS]
    + ['deepstack_0', 'deepstack_1', 'deepstack_2']
    + [f'decoder_layer_{i:02d}' for i in range(4)]
)
LOG = logging.getLogger('paired_page_extract')


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def pool_tensor(tensor):
    """Mean and population std over visual tokens; returns a compact float32 vector."""
    if not isinstance(tensor, torch.Tensor): return None
    x = tensor.detach()
    if x.ndim == 1: x = x.unsqueeze(0)
    x = x.reshape(-1, x.shape[-1]).float()
    return torch.cat((x.mean(0), x.std(0, unbiased=False))).cpu().numpy().astype(np.float32)


def resolve_image(value, data_root):
    p = Path(str(value))
    if p.is_file(): return p.resolve()
    p2 = data_root / p
    if p2.is_file(): return p2.resolve()
    # Manifest values are sometimes basenames; infer split folder from parent manifest.
    for split in ('train', 'val', 'test'):
        candidate = data_root / 'df_model' / split / p.name
        if candidate.is_file(): return candidate.resolve()
    raise FileNotFoundError(f'Cannot resolve image path from manifest value: {value!r}')


def output_tensors(output):
    """Flatten known Qwen vision output tensors, including DeepStack list members."""
    found = {}
    if hasattr(output, 'items'):
        for name, value in output.items():
            if isinstance(value, torch.Tensor): found[str(name)] = value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor): found[f'{name}_{i}'] = item
    # Explicit attribute covers ModelOutput implementations that omit lists in items().
    value = getattr(output, 'deepstack_features', None)
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            if isinstance(item, torch.Tensor): found[f'deepstack_features_{i}'] = item
    return found


def find_language_layers(model):
    """Return the decoder layer list for common Qwen3-VL model wrappers."""
    candidates = [
        getattr(getattr(model, 'model', None), 'language_model', None),
        getattr(model, 'language_model', None),
    ]
    for candidate in candidates:
        layers = getattr(getattr(candidate, 'layers', None), '__len__', None)
        if layers is not None:
            return candidate.layers
        inner = getattr(candidate, 'model', None)
        inner_layers = getattr(inner, 'layers', None)
        if inner_layers is not None:
            return inner_layers
    raise RuntimeError(
        'Could not locate Qwen language decoder layers. '
        'Inspect model.model.language_model in this Transformers version.'
    )


def select_decoder_indices(num_layers):
    """Choose four approximately evenly spaced decoder layers, 0-based."""
    if num_layers < 4:
        raise RuntimeError(f'Expected at least 4 decoder layers, found {num_layers}')
    indices = []
    for fraction in SELECTED_DECODER_LAYER_FRACTIONS:
        idx = min(num_layers - 1, max(0, round(fraction * (num_layers - 1))))
        if idx not in indices:
            indices.append(idx)
    if len(indices) != 4:
        # Deterministic fallback for unusual shallow/depth rounding cases.
        indices = [round(i * (num_layers - 1) / 3) for i in range(4)]
    return tuple(indices)


def extract_one(model, processor, image_path, outdir, doc_id, side, max_new_tokens):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    trans_path = outdir / 'transcription.txt'; stats_path = outdir / 'token_statistics.json'
    feat_path = outdir / 'compact_features.pt'; meta_path = outdir / 'page_metadata.json'
    if all(p.exists() for p in (trans_path, stats_path, feat_path, meta_path)):
        try:
            old = json.loads(meta_path.read_text(encoding='utf-8'))
            dimensions = old.get('feature_dimensions', {})
            saved_features = torch.load(feat_path, map_location='cpu', weights_only=True)
            saved_keys = set(saved_features.keys()) if isinstance(saved_features, dict) else set()
            arrays_valid = all(
                key in saved_keys
                and isinstance(saved_features[key], torch.Tensor)
                and saved_features[key].ndim == 1
                and saved_features[key].numel() > 0
                and torch.isfinite(saved_features[key]).all().item()
                for key in EXPECTED_FEATURE_KEYS
            )
            resume_matches = (
                old.get('status') == 'success'
                and old.get('image_path') == str(image_path)
                and old.get('max_new_tokens') == max_new_tokens
                and old.get('model_id') == MODEL_ID
                and old.get('selected_vision_blocks') == list(SELECTED_BLOCKS)
                and old.get('extraction_schema_version') == EXTRACTION_SCHEMA_VERSION
                and old.get('selected_decoder_blocks') is not None
                and all(key in dimensions for key in EXPECTED_FEATURE_KEYS)
                and arrays_valid
            )
            if resume_matches:
                LOG.info('Resume: skip %s%s (configuration and feature files validated)', doc_id, side)
                return old
            LOG.info('Resume: re-extract %s%s (saved outputs/configuration failed validation)', doc_id, side)
        except Exception as e:
            LOG.warning('Resume validation failed for %s%s; re-extracting: %s', doc_id, side, e)

    started = time.time(); pooled = {}; decoder_step_states = {}; hooks = []; hook_errors = []
    meta = {'document_id': doc_id, 'side': side, 'image_path': str(image_path),
            'model_id': MODEL_ID, 'max_new_tokens': max_new_tokens,
            'selected_vision_blocks': list(SELECTED_BLOCKS),
            'extraction_schema_version': EXTRACTION_SCHEMA_VERSION,
            'status': 'running'}
    try:
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        meta['image'] = {'width': image.width, 'height': image.height, 'mode': image.mode}
        visual = model.model.visual

        def block_hook(idx):
            def hook(module, inputs, output):
                try:
                    value = output[0] if isinstance(output, tuple) else output
                    pooled[f'vision_block_{idx:02d}'] = pool_tensor(value)
                except Exception as e: hook_errors.append(f'vision_block_{idx}: {type(e).__name__}: {e}')
            return hook
        for idx in SELECTED_BLOCKS:
            if idx >= len(visual.blocks): raise RuntimeError(f'Vision block {idx} missing; model has {len(visual.blocks)}')
            hooks.append(visual.blocks[idx].register_forward_hook(block_hook(idx)))

        def visual_hook(module, inputs, output):
            try:
                for name, tensor in output_tensors(output).items():
                    if 'deepstack_features_' in name.lower():
                        idx = name.rsplit('_', 1)[-1]
                        pooled[f'deepstack_{idx}'] = pool_tensor(tensor)
            except Exception as e: hook_errors.append(f'vision_output: {type(e).__name__}: {e}')
        hooks.append(visual.register_forward_hook(visual_hook))

        decoder_layers = find_language_layers(model)
        decoder_indices = select_decoder_indices(len(decoder_layers))
        meta['selected_decoder_blocks'] = list(decoder_indices)

        def decoder_hook(idx):
            # During generate(), the first call is the prompt prefill and later
            # calls are autoregressive decode steps. Keep only the final-position
            # hidden state from each call (the context used to predict the next token).
            # Pool the resulting token-step vectors after generation; never overwrite
            # earlier steps and never retain full prompt activations.
            def hook(module, inputs, output):
                try:
                    value = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(value, torch.Tensor) or value.ndim != 3:
                        raise RuntimeError(f'Unexpected decoder output shape: {getattr(value, "shape", None)}')
                    step_vector = value[:, -1, :].detach().float().cpu().squeeze(0)
                    decoder_step_states.setdefault(idx, []).append(step_vector)
                except Exception as e:
                    hook_errors.append(f'decoder_layer_{idx}: {type(e).__name__}: {e}')
            return hook

        for idx in decoder_indices:
            hooks.append(decoder_layers[idx].register_forward_hook(decoder_hook(idx)))

        messages = [{'role':'user','content':[
            {'type':'image','url':str(image_path)}, {'type':'text','text':PROMPT}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt')
        inputs.pop('token_type_ids', None)
        inputs = move_inputs_to_model_device(inputs, model)
        meta['input_tensors'] = {k:tensor_info(v) for k,v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=max_new_tokens,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                output_hidden_states=False)
        input_len = inputs['input_ids'].shape[1]
        generated = result.sequences[:, input_len:]
        token_count = int(generated.shape[1])
        transcription = processor.batch_decode(generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        eos = model.generation_config.eos_token_id
        eos_ids = set() if eos is None else ({eos} if isinstance(eos, int) else set(eos))
        last_id = int(generated[0, -1].item()) if token_count else None
        reached_limit = token_count >= max_new_tokens and last_id not in eos_ids
        token_stats = compute_token_uncertainty(result.scores or ())
        # Summarize the generation-context state sequence for each selected decoder
        # layer. Each call contributes the final-position state used to predict the
        # next token; mean/std pooling is across generation steps, not across layers.
        decoder_keys = [f'decoder_layer_{slot:02d}' for slot in range(len(decoder_indices))]
        score_steps = len(result.scores or ())
        for slot, idx in enumerate(decoder_indices):
            states = decoder_step_states.get(idx, [])
            if not states:
                raise RuntimeError(f'No generation-context states captured for decoder layer {idx}')
            if score_steps and len(states) != score_steps:
                raise RuntimeError(
                    f'Decoder capture/token-score mismatch for layer {idx}: '
                    f'{len(states)} forward states vs {score_steps} generation score steps'
                )
            pooled[f'decoder_layer_{slot:02d}'] = pool_tensor(torch.stack(states, dim=0))
        vision_keys = [f'vision_block_{i:02d}' for i in SELECTED_BLOCKS]

        probs = token_stats.get('token_probability', []); ent = token_stats.get('entropy', [])
        margins = token_stats.get('top1_top2_margin', [])
        gen_features = {
            'token_count': token_count,
            'mean_token_probability': float(np.mean(probs)) if probs else None,
            'std_token_probability': float(np.std(probs)) if probs else None,
            'min_token_probability': float(np.min(probs)) if probs else None,
            'mean_entropy_nats': float(np.mean(ent)) if ent else None,
            'std_entropy_nats': float(np.std(ent)) if ent else None,
            'max_entropy_nats': float(np.max(ent)) if ent else None,
            'mean_top1_top2_margin': float(np.mean(margins)) if margins else None,
            'min_top1_top2_margin': float(np.min(margins)) if margins else None,
            'reached_max_new_tokens': bool(reached_limit),
        }
        generation_score_steps = score_steps
        del generated, result, inputs
        cleanup_cuda()
        missing = [f'vision_block_{i:02d}' for i in SELECTED_BLOCKS if f'vision_block_{i:02d}' not in pooled]
        if missing: raise RuntimeError(f'Missing pooled vision features: {missing}')
        missing_expected = [k for k in EXPECTED_FEATURE_KEYS if k not in pooled or pooled[k] is None]
        if missing_expected:
            raise RuntimeError(f'Missing expected pooled features: {missing_expected}')
        if hook_errors: raise RuntimeError('; '.join(hook_errors))
        # Save each page's named feature vectors as CPU float32 tensors.
        page_features = {
            k: torch.as_tensor(v, dtype=torch.float32, device='cpu').contiguous()
            for k, v in pooled.items() if v is not None
        }
        torch.save(page_features, feat_path)
        trans_path.write_text(transcription, encoding='utf-8')
        write_json(stats_path, token_stats)
        meta.update({'status':'success', 'generation': {'generated_token_count':token_count,
            'transcription_characters':len(transcription), 'reached_max_new_tokens':bool(reached_limit),
            'last_generated_token_id':last_id,
            'decoder_forward_state_counts':{str(i):len(decoder_step_states.get(i, [])) for i in decoder_indices},
            'generation_score_steps':generation_score_steps}, 'generation_features':gen_features,
            'feature_dimensions':{k:int(v.shape[0]) for k,v in pooled.items() if v is not None},
            'hook_errors':hook_errors, 'gpu_memory':gpu_memory_info(),
            'duration_seconds':round(time.time()-started, 3)})
        write_json(meta_path, meta)
        LOG.info('Done %s%s: tokens=%d chars=%d duration=%.1fs', doc_id, side, token_count, len(transcription), meta['duration_seconds'])
        return meta
    except Exception as e:
        meta.update({'status':'failed','error':f'{type(e).__name__}: {e}',
                     'traceback':traceback.format_exc(),'hook_errors':hook_errors,
                     'duration_seconds':round(time.time()-started,3)})
        write_json(meta_path, meta); LOG.exception('Failed %s%s', doc_id, side); raise
    finally:
        for h in hooks:
            try: h.remove()
            except Exception: pass
        cleanup_cuda()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', choices=['train','val','test'], default='test')
    ap.add_argument('--limit', type=int, default=3, help='Documents; 0 means all selected rows')
    ap.add_argument('--start', type=int, default=0, help='Zero-based manifest row offset')
    ap.add_argument('--document-ids', nargs='*')
    ap.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    ap.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument('--output-root', type=Path, default=Path('outputs/paired_pilot'))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.max_new_tokens < 1: ap.error('--max-new-tokens must be positive')
    if args.start < 0: ap.error('--start must be non-negative')
    if args.limit < 0: ap.error('--limit must be non-negative (use 0 for all selected rows)')
    manifest = Path(f'outputs/manifests/{args.split}_manifest.csv')
    df = pd.read_csv(manifest, dtype={'document_id':str})
    for col in ('document_id','page_a','page_b'):
        if col not in df: raise ValueError(f'Missing manifest column: {col}')
    if args.document_ids:
        df = df[df.document_id.astype(str).isin(args.document_ids)]
    else: df = df.iloc[args.start:]
    if args.limit > 0: df = df.iloc[:args.limit]
    if df.empty: raise SystemExit('No documents selected; check split/limit/start/IDs')
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_config_path = args.output_root / 'run_config.json'
    run_config = {
        'split': args.split,
        'manifest': str(manifest),
        'model_id': MODEL_ID,
        'max_new_tokens': args.max_new_tokens,
        'selected_vision_blocks': list(SELECTED_BLOCKS),
        'selected_decoder_layer_fractions': list(SELECTED_DECODER_LAYER_FRACTIONS),
        'extraction_schema_version': EXTRACTION_SCHEMA_VERSION,
        'normalization_profile_for_evaluation': 'numbered_v1',
        'document_ids': df.document_id.astype(str).tolist(),
    }
    if run_config_path.exists():
        try:
            previous_config = json.loads(run_config_path.read_text(encoding='utf-8'))
        except Exception as e:
            raise SystemExit(f'Cannot read existing run config {run_config_path}: {e}')
        compatibility_keys = (
            'split', 'model_id', 'max_new_tokens', 'selected_vision_blocks',
            'selected_decoder_layer_fractions', 'extraction_schema_version',
        )
        mismatches = [
            key for key in compatibility_keys
            if previous_config.get(key) != run_config.get(key)
        ]
        if mismatches:
            raise SystemExit(
                f'Output directory {args.output_root} already has incompatible run configuration '
                f'({", ".join(mismatches)}). Choose a new --output-root or archive the old run.'
            )
    write_json(run_config_path, run_config)
    LOG.info('Loading model once for %d documents / %d pages', len(df), 2*len(df))
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype,
        device_map=DEVICE_MAP, low_cpu_mem_usage=True, attn_implementation=ATTENTION_IMPLEMENTATION)
    model.eval(); statuses=[]
    for _, row in df.iterrows():
        doc_id = str(row['document_id']); doc_dir = args.output_root/'pages'/doc_id
        for side, col in [('A','page_a'),('B','page_b')]:
            try:
                image_path = resolve_image(row[col], args.data_root)
                meta = extract_one(model, processor, image_path, doc_dir/side, doc_id, side, args.max_new_tokens)
                statuses.append({'document_id':doc_id,'side':side,'status':meta['status'],
                    'generated_token_count':meta.get('generation',{}).get('generated_token_count'),
                    'reached_max_new_tokens':meta.get('generation',{}).get('reached_max_new_tokens'),
                    'duration_seconds':meta.get('duration_seconds')})
            except Exception as e:
                statuses.append({'document_id':doc_id,'side':side,'status':'failed','error':f'{type(e).__name__}: {e}'})
            write_json(args.output_root/'page_status.json', statuses)
        a = doc_dir/'A'/'transcription.txt'; b = doc_dir/'B'/'transcription.txt'
        if a.exists() and b.exists():
            merged = a.read_text(encoding='utf-8').strip('\r\n')+'\n\n'+b.read_text(encoding='utf-8').strip('\r\n')
            pred_dir = args.output_root/'prediction_pages'; pred_dir.mkdir(parents=True, exist_ok=True)
            (pred_dir/f'{doc_id}A.txt').write_text(a.read_text(encoding='utf-8'),encoding='utf-8')
            (pred_dir/f'{doc_id}B.txt').write_text(b.read_text(encoding='utf-8'),encoding='utf-8')
            merged_dir = args.output_root/'merged_predictions'; merged_dir.mkdir(parents=True, exist_ok=True)
            (merged_dir/f'{doc_id}.txt').write_text(merged.strip('\r\n'),encoding='utf-8')
            # Save both page feature dictionaries plus a concatenated paired vector.
            # The combined vector uses explicit A__/B__ prefixes to retain page identity.
            feature_a_path = doc_dir/'A'/'compact_features.pt'
            feature_b_path = doc_dir/'B'/'compact_features.pt'
            if feature_a_path.exists() and feature_b_path.exists():
                try:
                    features_a = torch.load(feature_a_path, map_location='cpu', weights_only=True)
                    features_b = torch.load(feature_b_path, map_location='cpu', weights_only=True)
                    paired = {}
                    for key, value in features_a.items():
                        paired[f'A__{key}'] = value.detach().cpu().float().contiguous()
                    for key, value in features_b.items():
                        paired[f'B__{key}'] = value.detach().cpu().float().contiguous()
                    # One flat vector for conventional document-level ML classifiers.
                    paired['paired_concat'] = torch.cat(
                        [paired[f'A__{key}'].flatten() for key in sorted(features_a)] +
                        [paired[f'B__{key}'].flatten() for key in sorted(features_b)]
                    )
                    paired['document_id'] = doc_id
                    torch.save(paired, doc_dir/'paired_features.pt')
                    LOG.info('Saved paired features: %s', doc_dir/'paired_features.pt')
                except Exception as e:
                    LOG.exception('Could not create paired feature file for %s: %s', doc_id, e)
    # Stable compact feature table: one row per page; vectors stay in per-page PT files.
    rows=[]
    for _, row in df.iterrows():
        doc_id=str(row['document_id'])
        for side in ('A','B'):
            md=args.output_root/'pages'/doc_id/side/'page_metadata.json'
            fp=args.output_root/'pages'/doc_id/side/'compact_features.pt'
            if md.exists() and fp.exists():
                m=json.loads(md.read_text(encoding='utf-8'))
                rows.append({'document_id':doc_id,'side':side,'status':m.get('status'),
                    'feature_file':str(fp),'metadata_file':str(md),
                    **{f'gen_{k}':v for k,v in m.get('generation_features',{}).items()}})
    pd.DataFrame(rows).to_csv(args.output_root/'page_feature_index.csv',index=False)
    LOG.info('Finished. Inspect %s/page_status.json and page_feature_index.csv', args.output_root)

if __name__ == '__main__': main()
