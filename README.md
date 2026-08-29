# Cross-border Material Agent

Python 3.12 agent for the Qwen AI Arena cross-border e-commerce localization task.

## Runtime

The evaluator invokes:

```bash
python agent.py --prompt "...input path...output path..."
```

The agent discovers the product, category and attribute JSON files by schema; builds an evidence-backed fact ledger; resolves the AliExpress leaf category and enumerated attributes; creates English, Korean and Brazilian Portuguese descriptions; produces one main image, five deliberately distinct detail images and one video; then evaluates and validates the complete delivery before exiting.

The runtime is a bounded agent rather than an unconstrained shell agent. An LLM manager first plans creative emphasis, locale priorities and rubric risks while the executor enforces a fixed 95-point internal acceptance score. After source-image inspection, two independent evidence reconcilers resolve structured-versus-visual appearance conflicts through a generic indexed fact ledger, without product-specific exceptions. Final delivery refinement uses separated roles: independent evaluators report evidence-backed defects without scores or tool calls; an adjudicator resolves only disputed findings; code derives A1-A7 scores from accepted findings; a repair planner maps accepted defect IDs to the bounded tool catalog; tools execute reversible target transactions; and a separate verifier checks the requested postcondition and critical regressions. The controller caches evaluation by an exact artifact fingerprint, stops immediately when ready, never re-evaluates unchanged output, and permits at most one same-fingerprint replan after a rejected or no-op repair. Final generated assets are reviewed through their model-result URLs; locally cropped or deterministically rendered files are represented by their final hashes, physical inspection and exact deterministic render evidence, never by a provenance image masquerading as final pixels. Every successful tool call must change the target content hash, synchronize dependencies, pass local consistency checks, and satisfy the outcome verifier before it is committed. Each locale copy and generated-media target is an independent transaction; rejected candidates restore only their pre-action checkpoint.

In full mode, taxonomy resolution uses two bounded ReAct tasks rather than a pre-ranked candidate classifier: the first selects a leaf category and the second receives that result in a fresh context to resolve its attribute schema and mappings. Each task has its own independent 50-turn budget; neither task consumes or inherits the other's turns. Every turn exposes that task's remaining budget, and the last two turns of each task expose only its validated finish tool so one rejected finish can still be corrected. Like a constrained Pi-style file explorer, the chat model gets two generic read-only tools instead of task-specific search rules: a batched query over normalized `categories`, `schemas`, `attributes` and `values` collections, plus exact `ref` reads with pagination. The model chooses filters, fields, AND/OR semantics and follow-up queries; responses are capped at 50 KB and are never semantically ranked by code. Each tool observation is returned as a `role=tool` message, while code validates the selected leaf plus every submitted schema/attribute/value ID and exact source attribute pair without choosing the semantic winner. The older lexical scorer remains only as an offline or API-failure fallback and is not included as an online model hint.

Localized copy uses a writer pass followed by a factual-auditor pass and locale guards for en-US, ko-KR and pt-BR. The models author only the buyer-facing title, overview, highlights and fit note; code independently builds the machine appendix from verified taxonomy, attribute, SKU and media contracts, so prose revisions cannot mutate parseable fields. Every description leads with two substantive shopper-facing paragraphs and natural feature highlights, followed by compact localized listing tables. Category IDs, platform attribute/value IDs, SKU IDs and Spec IDs remain parseable; localized source labels distinguish seller facts from platform mappings, while raw Chinese values and JSON pointers stay in the private fact ledger. Legible source-image size charts are transcribed conservatively, checked against SKU labels, and deterministically converted into localized measurement tables and a clean detail image. Unsupported numeric fields are repaired independently so one bad number no longer discards an otherwise valid localized draft.

Every inspected source image receives a role and risk record covering intrinsic product print versus marketing overlays, watermarks, contact details, QR codes, prices, platform marks and sensitive visuals. Unsafe references are excluded from derivative generation. The hero uses three generated candidates when safe references exist, then selects by product identity, construction, color, completeness, clean background and usability. Source-image hero fallback separately rejects people, unrelated props, multiple products, cropping and lifestyle backgrounds whenever an inspected listing-ready source exists. Detail references are selected across evenly sampled product, SKU-variant and description images by storyboard role; every generative slot uses two candidates and first passes a per-slot hard gate. The surviving candidate pool is then judged together with the fixed hero, and a different combination is installed atomically only when it improves the six-image set by at least three points without introducing a hard defect. Generated video uses category-aware motion constraints, is fully decoded, and is silent by default. A deterministic source image or multi-shot catalog video is used only when initial generation is unavailable or yields no acceptable candidate, not as a response to later evaluator feedback.

Task-specific skills are shipped under `crossborder_agent/skills/` and progressively loaded into the relevant model calls: delivery planning, product grounding, AliExpress taxonomy, AliExpress content compliance, marketplace localization, commerce visuals, commerce video, and the A1-A7 rubric evaluator. They are local prompt-policy packages; taxonomy exploration uses the agent's own bounded function tools over the supplied local snapshots and does not rely on MCP, an external vector service, or an external taxonomy API.

Required environment variables:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `OPENAI_BASE_URL`
- `AGENT_LOG_DIR`

A normal run fails fast when any model endpoint variable is absent, instead of silently producing a model-free submission. `--offline` is the explicit development-only escape hatch.

Optional model overrides:

- `AGENT_CHAT_MODEL`
- `AGENT_CHAT_FALLBACK_MODEL`
- `AGENT_REVIEW_MODEL`
- `AGENT_REVIEW_FALLBACK_MODEL`
- `AGENT_EVALUATION_MODELS` (comma-separated list of 2 or 3 evaluator models)
- `AGENT_IMAGE_MODEL`
- `AGENT_IMAGE_FALLBACK_MODEL`
- `AGENT_VIDEO_MODEL`
- `AGENT_KEEP_VIDEO_AUDIO=1` to retain source audio explicitly; the default removes unreviewed audio

Defaults are `qwen3.8-max` for planning and localization, `qwen3.8-max,qwen3.7-plus` for independent evidence-finding, `wan2.7-image-pro` for source-guided image generation, `qwen-image-3.0-pro` as the cross-model image fallback, and `wan2.7-i2v-2026-04-25` for video.

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

`--offline` skips model calls but still downloads supplied source URLs, produces localized fallback copy, normalizes images, creates a playable video, and executes all delivery gates. Distinct seller views are exhausted first; if too few remain, bounded neckline/hem crops are generated and rechecked against all accepted image hashes before the delivery is installed.

## Build the submission

```bash
python scripts/build_submission.py
```

This downloads pinned CPython 3.12 manylinux x86_64 wheels, vendors Pillow and the ffmpeg fallback runtime, and creates `dist/agent.zip`. The ZIP has `agent.py` at its root and is rejected by the build script if it exceeds 100 MB.
