# Cross-border Material Agent

Python 3.12 agent for the Qwen AI Arena cross-border e-commerce localization task.

## Runtime

The evaluator invokes:

```bash
python agent.py --prompt "...input path...output path..."
```

The agent discovers the product, category and attribute JSON files by schema; builds an evidence-backed fact ledger; resolves the AliExpress leaf category and enumerated attributes; creates English, Korean and Brazilian Portuguese descriptions; produces one main image, five deliberately distinct detail images and one video; then evaluates and validates the complete delivery before exiting.

The runtime is organized around a versioned decision state, not a growing list of product rules. It first collects structured, SKU, taxonomy-snapshot and per-image visual evidence. Independent model reconcilers turn conflicts into a canonical product state before taxonomy is selected. Taxonomy, publishable claims, localization and visual plans are projections of that state; a frozen expected-delivery specification records the accepted claim, mapping, locale, file and visual-evidence coverage before production begins. The top-level native tool-calling orchestrator then chooses the storyboard, exact safe source-image indexes for each job, candidate breadth, locale priorities and production order. After production it can request independent review, invoke reversible repairs, reopen evidence or taxonomy decisions, and decide when to finish. Repairs update the upstream version, rebuild affected projections, invalidate downstream review, and are committed atomically only after the current fingerprint is reviewed. Host code enforces protocol, provenance, safety, dependency freshness and file integrity, while models retain the open-ended semantic decisions.

In full mode, taxonomy resolution uses two bounded ReAct tasks rather than a pre-ranked candidate classifier: the first selects a leaf category and the second receives that result in a fresh context to resolve its attribute schema and mappings. Each task has its own independent 50-turn budget; neither task consumes or inherits the other's turns. Every turn exposes that task's remaining budget, and the last two turns of each task expose only its validated finish tool so one rejected finish can still be corrected. Like a constrained Pi-style file explorer, the chat model gets two generic read-only tools instead of task-specific search rules: a batched query over normalized `categories`, `schemas`, `attributes` and `values` collections, plus exact `ref` reads with pagination. The model chooses filters, fields, AND/OR semantics and follow-up queries; responses are capped at 50 KB and are never semantically ranked by code. Each tool observation is returned as a `role=tool` message, while code validates the selected leaf plus every submitted schema/attribute/value ID and exact source attribute pair without choosing the semantic winner. The older lexical scorer remains only as an offline or API-failure fallback and is not included as an online model hint.

Localized copy uses a writer pass followed by a factual-auditor pass and locale guards for en-US, ko-KR and pt-BR. The models author only the buyer-facing title, overview, highlights and fit note; code independently builds the machine appendix from verified taxonomy, attribute, SKU and media contracts, so prose revisions cannot mutate parseable fields. Every description leads with two substantive shopper-facing paragraphs and natural feature highlights, followed by compact localized listing tables. Category IDs, platform attribute/value IDs, SKU IDs and Spec IDs remain parseable; localized source labels distinguish seller facts from platform mappings, while raw Chinese values and JSON pointers stay in the private fact ledger. Legible source-image size charts are transcribed conservatively, checked against SKU labels, and deterministically converted into localized measurement tables and a clean detail image. Unsupported numeric fields are repaired independently so one bad number no longer discards an otherwise valid localized draft.

Every inspected source image receives a role and risk record covering intrinsic product print versus marketing overlays, watermarks, contact details, QR codes, prices, platform marks and sensitive visuals. Unsafe references are excluded from derivative generation. The orchestrator chooses candidate counts from one to four and prioritizes source-image roles separately for the hero and every detail job. Candidate selectors enforce product identity, construction, color and hard usability constraints without redefining the model's storyboard. The complete surviving candidate state—including the actual current candidate and local-review evidence—is reviewed as a six-image set; it is never truncated by generation order or repaired by substituting a fabricated current index. The set may be changed again through the top-level repair loop. Generated video is fully decoded and silent by default. Deterministic source images and catalog video remain emergency completeness fallbacks when a generation capability is unavailable or produces no valid artifact.

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

The automated suite focuses on complete protocol smoke runs, deterministic tool boundaries, safety/data-integrity contracts and structurally valid delivery output. Semantic quality and generalization belong in holdout evals rather than unit tests that freeze one model trajectory, prompt wording, call count or sample answer. Resilience tests cover rate limits, terminal client errors, malformed JSON, corrupt media, failed asynchronous tasks and deadline-aware fallback.

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
