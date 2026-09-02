# Cross-border Material Agent

**Turn seller product data into multilingual listing copy, product images, and video.**

Cross-border Material Agent is a Python agent for cross-border e-commerce content production. It combines seller facts, SKU data, source images, and local AliExpress taxonomy snapshots to produce a coordinated set of listing materials in English, Korean, and Brazilian Portuguese.

Built for the Qwen AI Arena cross-border e-commerce localization task, the project provides a CLI and a modular pipeline for exploring evidence-grounded, multimodal agent workflows. Its current data adapters and bundled examples focus on apparel.

[Quick start](#quick-start) · [Configuration](#configuration) · [Input data](#input-data) · [How it works](#how-it-works) · [Contributing](#contributing)

## Features

- **Evidence-grounded content.** Reconciles structured product fields, SKU information, and visual observations before deciding which claims belong in copy, media, and marketplace mappings.
- **Local taxonomy exploration.** Resolves an AliExpress leaf category and its attributes using bounded tools over supplied JSON snapshots, with provenance for accepted mappings.
- **Three target markets.** Writes and audits copy for US English (`en-US`), South Korean (`ko-KR`), and Brazilian Portuguese (`pt-BR`) audiences while preserving listing identifiers in a machine-readable appendix.
- **Coordinated media production.** Plans a hero image, five detail images, and a product video; inspects source-image risks and reviews generated candidates for product consistency.
- **Review and repair.** Uses independent model reviews to guide revisions, with versioned decisions and reversible artifact updates.
- **Resilient delivery.** Applies bounded retries, capability fallbacks, and local media construction when model services fail or time runs short.

## Output

Each run targets the following **11 files**:

| Artifact | Filename | Purpose |
| --- | --- | --- |
| English description | `product_description_en.md` | US English listing copy |
| Korean description | `product_description_ko.md` | South Korean listing copy |
| Portuguese description | `product_description_pt.md` | Brazilian Portuguese listing copy |
| Main image | `main_image.jpeg` | Hero image, at least 800 × 800 pixels and below 1 MB |
| Detail images | `detail_image_1.jpeg` through `detail_image_5.jpeg` | Five complementary product views or explanations |
| Product video | `product_video.mp4` | Product presentation; audio removed by default |
| Strategy document | `strategy_document.md` | English explanation of grounding, localization, and media choices |

Descriptions include shopper-facing prose and structured listing details: source platform, product ID and URL, taxonomy and attribute identifiers, SKU breakdowns, and a media guide. Operational logs and the strategy document are in English; exact source values may remain in the machine appendix when a trustworthy translation is unavailable.

## Quick start

### 1. Install

Use **Python 3.12**. The submission build targets Debian 12 on x86_64; the commands below use a POSIX shell.

```bash
git clone https://github.com/jastfkjg/cross-border-material-agent.git
cd cross-border-material-agent

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Runtime dependencies are pinned in [requirements.txt](requirements.txt): Pillow for images and `imageio-ffmpeg` for video processing. The runtime uses the bundled FFmpeg executable when available, with system `ffmpeg` as a fallback. Noto Sans CJK is bundled for multilingual table rendering.

For the agent's optional command-based inspection tools, make `rg`, `jq`, `find`, `file`, and `ffprobe` available on `PATH`. Native file listing, reading, and search tools are also provided.

### 2. Try the bundled data without model calls

The bundled dataset contains multiple products, so select one with `--product-id`:

```bash
python agent.py \
  --input-dir Data_for_Users \
  --output-dir output/offline-demo \
  --product-id 3887087154767 \
  --offline
```

Open `output/offline-demo/` to inspect the delivery. Logs are written to `.agent-logs/agent.log` by default.

**`--offline` disables model API calls, but still downloads source images over HTTP(S).** It exercises fallback copy, image processing, video construction, and delivery validation. It is useful for checking the workflow; it does not demonstrate the semantic quality of a full model-assisted run. For a smoke test using locally served images, see [Development and testing](#development-and-testing).

### 3. Run with model services

Set the three required API variables in your shell, replacing the placeholders with credentials and endpoints for your provider:

```bash
export DASHSCOPE_API_KEY="<your-api-key>"
export DASHSCOPE_BASE_URL="<your-native-dashscope-api-base-url>"
export OPENAI_BASE_URL="<your-openai-compatible-api-base-url>"

python agent.py \
  --input-dir Data_for_Users \
  --output-dir output/model-demo \
  --product-id 3887087154767 \
  --run-profile full
```

The adapter uses OpenAI-compatible chat completions and native DashScope image, video, and task APIs, authenticated with the same `DASHSCOPE_API_KEY`. A chat endpoint alone is insufficient. Configure model names supported by your endpoints using the overrides below. Model-assisted runs send product evidence and images to the configured services and consume provider quota.

If any required API variable is missing, the agent logs the configuration problem and switches to deterministic fallback. A successful exit can therefore include fallback artifacts; check the logs to understand which capabilities ran.

## Configuration

### API and runtime settings

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | Unset | Shared credential for chat and native media APIs |
| `DASHSCOPE_BASE_URL` | Unset | Native DashScope API base URL |
| `OPENAI_BASE_URL` | Unset | OpenAI-compatible chat API base URL |
| `AGENT_LOG_DIR` | `.agent-logs` | Directory for `agent.log` and optional debug traces |
| `AGENT_RUN_PROFILE` | `full` | Default profile; overridden by `--run-profile` |
| `AGENT_DEBUG` | Disabled | Set to `1` to enable structured debug traces, or use `--debug` |
| `AGENT_KEEP_VIDEO_AUDIO` | Disabled | Set to `1` to retain generated video audio |

These settings are read from the process environment; the CLI does not automatically load a `.env` file. Keep the log directory separate from the output directory when you need exactly the 11 delivery files.

<details>
<summary>Model overrides and defaults</summary>

The following are defaults in this repository, not a guarantee of availability on every provider account.

| Environment variable | Default |
| --- | --- |
| `AGENT_CHAT_MODEL` | `qwen3.8-max` |
| `AGENT_CHAT_FALLBACK_MODEL` | `qwen3.7-plus` |
| `AGENT_REVIEW_MODEL` | `qwen3.8-max` |
| `AGENT_REVIEW_FALLBACK_MODEL` | `qwen3.7-plus` |
| `AGENT_VISUAL_REVIEW_MODEL` | Value of `AGENT_REVIEW_MODEL` |
| `AGENT_VISUAL_REVIEW_FALLBACK_MODEL` | Value of `AGENT_REVIEW_FALLBACK_MODEL` |
| `AGENT_EVALUATION_MODELS` | `qwen3.8-max,qwen3.7-plus` |
| `AGENT_IMAGE_MODEL` | `wan2.7-image-pro` |
| `AGENT_IMAGE_FALLBACK_MODEL` | `qwen-image-3.0-pro` |
| `AGENT_VIDEO_MODEL` | `wan2.7-i2v-2026-04-25` |
| `AGENT_VIDEO_FALLBACK_MODELS` | `happyhorse-1.1-r2v,happyhorse-1.1-t2v` |

`AGENT_EVALUATION_MODELS` accepts two or three distinct comma-separated model names. `AGENT_VIDEO_FALLBACK_MODELS` is an ordered comma-separated list of up to three models. See [api.py](crossborder_agent/api.py) for the configuration and provider adapters.

</details>

### Command-line options

```bash
python agent.py --help
```

| Option | Behavior |
| --- | --- |
| `--input-dir PATH` | Input directory; must be paired with `--output-dir` |
| `--output-dir PATH` | Delivery directory; use a separate directory for each run |
| `--product-id ID` | Select one product from a multi-product dataset |
| `--prompt TEXT` | Read input and output paths from a task prompt when explicit directories are omitted |
| `--offline` | Skip model API calls and exercise fallback production |
| `--run-profile full` | Full model-assisted workflow when API configuration is present |
| `--run-profile fast` | Reduce model calls, skip full taxonomy adjudication, and use local video construction |
| `--timeout-seconds N` | Internal deadline in seconds; default `1740`, minimum `60` |
| `--debug` | Write redacted `TRACE_JSON` events to `agent_debug.jsonl` in the log directory |
| `--version` | Print the agent version and exit |

`fast` is intended for local iteration and can still call paid model services. Use `--offline` to disable those calls.

For evaluation harnesses, the prompt interface accepts labeled absolute paths:

```bash
python agent.py --prompt "Input directory: /absolute/path/to/input
Output directory: /absolute/path/to/output"
```

## Input data

The agent recursively discovers JSON files by their structure. Filenames can vary, but the data must follow the supported schemas. A normal input directory contains one product, one category snapshot, and one attribute snapshot:

```text
input/
├── product_info/
│   └── product.json
├── categories.json
└── attributes.json
```

| Input | Expected structure |
| --- | --- |
| Product | Seller payload under `ret.result.result`, including `offerId`, `subject` or `subjectTrans`, and at least one usable HTTP(S) image URL. Attributes, SKUs, and description images provide additional evidence. |
| Categories | A `categories` array containing category nodes with `catId`, `name`, `categoryPath`, `isLeaf`, and optional `children`. |
| Attributes | A `categories` array containing `categoryId` and `categoryMetadata`, including `categoryProductAttrList` and `categorySaleAttrList` definitions and their enumerated values. |

See [Data_for_Users](Data_for_Users) for complete examples and [input_loader.py](crossborder_agent/input_loader.py) for discovery and normalization. The bundled eleven products illustrate the input contract; they are not a benchmark of general performance. Select exactly one product per invocation, using `--product-id` when the directory contains more than one.

Taxonomy resolution uses the supplied snapshots locally. It does not fetch a live marketplace taxonomy, use an external vector database, or require an MCP server. Adapting a new seller export or marketplace requires compatible input adapters and taxonomy data.

## How it works

1. **Load and reconcile evidence.** Normalize seller fields, SKUs, and source images into a fact ledger. Model reconcilers resolve conflicting evidence into a versioned product state.
2. **Resolve marketplace mappings.** In full mode, separate bounded agent tasks select the leaf category and resolve its attribute schema. Accepted mappings must cite source evidence and valid taxonomy entries.
3. **Plan the delivery.** Freeze the expected claims, mappings, locales, and artifacts. The orchestrator chooses the visual narrative, source references, candidate counts, and production order.
4. **Produce copy and media.** Write and audit localized prose, build the identifier appendix from verified data, generate and select images, and create the video.
5. **Review, repair, and commit.** Independent reviewers identify evidence-backed defects. The orchestrator can revise upstream decisions or artifacts; deterministic validation checks the resulting delivery before commit and recovery paths restore missing or invalid files where possible.

The model owns semantic decisions such as interpreting evidence, choosing categories, and planning repairs. Host code enforces schemas, provenance, resource limits, and artifact integrity. Taxonomy tools expose bounded listing, reading, regex search, staging writes, and restricted commands; command execution uses an allowlist and runs without a shell or secret environment variables.

Semantic review findings guide improvement rather than acting as an availability gate. If review or generation services fail, the runtime attempts to deliver the best available complete artifact set using surviving candidates or source-based fallbacks. **Structural validity and a successful exit do not certify translation quality, visual fidelity, or marketplace acceptance.** Review generated materials before publishing.

## Project structure

```text
agent.py                        CLI entry point
agent.json                      Runtime and version manifest
crossborder_agent/
├── cli.py                      Arguments, logging, and final recovery boundary
├── pipeline.py                 Public Pipeline facade
├── pipeline_parts/             Evidence, taxonomy, planning, production,
│                               review, and transaction implementation
├── agent_loop.py               Native tool-calling execution loop
├── agent_workspace.py          Bounded file and command tools
├── taxonomy_agent.py           Category and attribute agent tasks
├── decision_state.py           Versioned decisions and delivery contracts
├── api.py                      Model configuration and HTTP adapters
├── localization.py             Localized copy and listing appendices
├── media.py                    Image and video utilities
├── qa.py                       Deterministic delivery validation
├── skills/                     Prompt-policy packages loaded by the agent
└── assets/fonts/               Bundled multilingual font and its license
rules/                          Marketplace content policy data
Data_for_Users/                  Sample products and taxonomy snapshots
scripts/                        Submission builder and API diagnostics
tests/                          Smoke, tool-boundary, and integrity tests
```

## Development and testing

After installing the runtime dependencies, run the existing offline smoke test:

```bash
python -m unittest discover -s tests -p 'test_offline_pipeline.py' -v
```

This test serves synthetic source images on localhost, runs a complete pipeline, and checks the 11-file delivery contract, including playable video. It does not require model credentials or external image downloads.

Run the full suite with the standard library test runner:

```bash
python -m unittest discover -s tests -v
```

Tests cover complete execution, model protocol handling with local fixtures, bounded tools, provenance, media integrity, and recovery behavior. Semantic quality and generalization should be evaluated separately on unseen inputs rather than by fixing one model trajectory or sample answer in unit tests.

For runtime diagnostics, add `--debug` and inspect `agent.log` alongside `agent_debug.jsonl`. When reporting an issue, include the command, Python version, run profile, and relevant redacted log excerpts. Remove credentials and private product data before sharing diagnostics.

## Build an evaluation submission

The original evaluation harness expects a self-contained ZIP with `agent.py` at its root:

```bash
python scripts/build_submission.py
```

The builder downloads pinned CPython 3.12 manylinux x86_64 wheels, vendors Pillow and the FFmpeg runtime, checks the manifest and bundled font, and creates `dist/agent.zip`. Archives larger than 100 MiB are rejected. This artifact targets Debian 12 x86_64 and is separate from the local virtual-environment installation.

For a source-only archive without dependency downloads:

```bash
python scripts/build_submission.py --skip-dependencies
```

The source-only archive requires dependencies in the target environment. Both commands write `dist/agent.zip`; the build includes only the entry point, manifest, requirements, agent package, and rules, excluding sample data and local outputs.

## Contributing

Issues and pull requests are welcome. Read [AGENTS.md](AGENTS.md) for the repository's development principles.

- Describe the problem and the externally observable behavior you want to improve.
- Keep changes applicable to unseen products, categories, attributes, and languages. Avoid routing rules derived from sample IDs, titles, or known answers.
- Improve model evidence, tool affordances, and validation feedback before adding semantic special cases to host code.
- Keep tests small and deterministic. Add a regression test when it protects a stable public contract or a strategy-independent failure mode.
- Include relevant validation results and update documentation when behavior or configuration changes.

## License

A project-wide license has not yet been added to this repository.

The bundled Noto Sans CJK font is distributed under the [SIL Open Font License 1.1](crossborder_agent/assets/fonts/OFL-1.1.txt); see the [font attribution](crossborder_agent/assets/fonts/README.md) for details.
