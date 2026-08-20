# Cross-border Material Agent

Python 3.12 agent for the Qwen AI Arena cross-border e-commerce localization task.

## Runtime

The evaluator invokes:

```bash
python agent.py --prompt "...input path...output path..."
```

The agent discovers the product, category and attribute JSON files by schema; builds an evidence-backed fact ledger; resolves the AliExpress leaf category and enumerated attributes; creates English, Korean and Brazilian Portuguese descriptions; produces one main image, five deliberately distinct detail images and one video; then validates the complete delivery before exiting.

Localized copy uses a writer pass followed by a factual-auditor pass and locale guards for en-US, ko-KR and pt-BR. Every description now carries both shopper-facing localized values and the exact source/category/SKU values with JSON evidence pointers, making factual and taxonomy fields directly machine-verifiable. Seller-provided weight guidance is tied to each exact SKU label before deterministic kg/lb conversion.

Every inspected source image receives a role and risk record covering visible text, watermarks, contact details, QR codes, prices, platform marks and suspected third-party branding. Unsafe references are excluded from derivative generation. The hero uses three generated candidates when safe references exist, then selects by product identity, construction, color, completeness, clean background and usability. Source-image hero fallback separately rejects people, unrelated props, multiple products, cropping and lifestyle backgrounds whenever an inspected listing-ready source exists. Detail references are selected across the full safe source set by storyboard role, and the set uses distinct US, Korean, Brazilian and cross-market visual treatments without generated text or stereotypes. Detail assets are checked for storyboard fit, color/pattern/structure drift, prohibited visuals, collage misuse and near-duplication. Generated video uses category-aware motion constraints, is fully decoded and semantically reviewed, and is silent by default; if generation fails, the final validated image set is assembled into a playable multi-shot H.264 catalog video.

Required environment variables:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `OPENAI_BASE_URL`
- `AGENT_LOG_DIR`

Optional model overrides:

- `AGENT_CHAT_MODEL`
- `AGENT_CHAT_FALLBACK_MODEL`
- `AGENT_IMAGE_MODEL`
- `AGENT_IMAGE_FALLBACK_MODEL`
- `AGENT_VIDEO_MODEL`
- `AGENT_KEEP_VIDEO_AUDIO=1` to retain source audio explicitly; the default removes unreviewed audio

Defaults are `qwen3.8-max` for planning, localization and multimodal QA, `wan2.7-image-pro` for source-guided image generation, `qwen-image-3.0-pro` as the cross-model image fallback, and `wan2.7-i2v-2026-04-25` for video.

The versioned local content policy distinguishes cited AliExpress platform obligations from conservative runtime guards. The sample taxonomy golden set freezes the expected category and selected key-attribute mappings for all eleven supplied records. Resilience tests cover rate limits, terminal client errors, malformed JSON, corrupt media, failed asynchronous tasks and deadline-aware fallback.

## Development smoke run

The input directory must contain exactly one product JSON. To use the supplied multi-product development dataset, select a product ID:

```bash
python agent.py \
  --input-dir Data_for_Users \
  --output-dir /tmp/material-agent-output \
  --product-id 3887087154767 \
  --offline
```

`--offline` skips model calls but still downloads supplied source URLs, produces localized fallback copy, normalizes images, creates a playable video, and executes all delivery gates.

## Build the submission

```bash
python scripts/build_submission.py
```

This downloads pinned CPython 3.12 manylinux x86_64 wheels, vendors Pillow and the ffmpeg fallback runtime, and creates `dist/agent.zip`. The ZIP has `agent.py` at its root and is rejected by the build script if it exceeds 100 MB.
