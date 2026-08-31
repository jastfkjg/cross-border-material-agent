Task Data
We has provided sample product data as well as category and attribute data; participants may use this data to design and build agents capable of generating content. This data is intended solely for use as examples within this task; participants are prohibited from using it for commercial purposes without authorization.

Data path: ./Data_for_Users

## I. Task Background
This task focuses on localized product asset generation: participants are required to build an AI Agent that automatically generates the complete set of localized assets needed to list a specified product on plateform in the target market.（Use AliExpress as an example.）

Taking a product from source data to a live listing on an overseas platform involves a substantial amount of localization work:

Language localization: producing product copy for different markets (e.g., the United States, South Korea, and Brazil) that follows local language conventions rather than relying on word-for-word translation.
Visual assets: producing a main image, product detail images, and a product video that meet platform specifications.
Platform adaptation: accurately mapping the product to the platform's category and attribute taxonomy.
Market adaptation: converting sizes, following the target market's spelling and phrasing conventions, and accounting for cultural sensitivities.
Factual consistency: ensuring that all information in every asset remains faithful to the source data, with nothing fabricated or added beyond what the source supports.
These steps have traditionally required operations, design, translation, and other teams to work together.

This task asks participants to build an Agent that combines all of these steps into a single automated run, delivering the full asset set end to end.

Evaluation follows a two-stage process: 「Automated Evaluation & Expert Review」All submitted Agents are run and evaluated under consistent conditions by the platform and expert panel.


## II. Task Guide
1.Note A: Submission Standards and Environment Configuration
Submission Standards

Participants must submit a complete Agent code package that the platform can run independently. Once the platform runs the code package, the Agent must automatically produce all assets to be evaluated. The code package must meet the following requirements:

Upload the deliverable as a ZIP archive (≤100 MB) containing an entry file named agent.py, agent.js, agent.jar, or agent (a Go-compiled binary with no file extension).
The package must include all runtime dependencies; the platform will automatically detect the runtime based on the file extension and launch the program.
At startup, the platform passes a natural-language instruction via the --prompt argument. Your program must parse this instruction to extract the input and output data paths, then complete the task.
On successful completion, the program must exit with exit code 0; on failure, it must return a non-zero exit code.
Environment Configuration

The runtime environment is Linux (Debian 12 x86_64), pre-installed with and limited to JDK 17, Python 3.12, Node.js 22, and Go 1.22. The platform automatically selects the runtime based on the entry file's extension (.py runs on Python, .js on Node.js, .jar on Java, and an extensionless ELF binary is treated as a Go-compiled binary), and passes a natural-language instruction containing the input and output paths via --prompt "..."; your program must parse the prompt to complete the task.

The specific details are as follows:

The deliverable directory must be named agent, and the package must be a ZIP archive with a compressed size no greater than 100 MB.T
he entry file must be located in the root directory, named agent.py , agent.js , agent.jar (with a declared Main-Class), or agent (a Go-compiled binary with no extension).
The root directory must contain an agent.json file in the format {"runtime":"name","version":"x.y.z"}. The runtime field is determined by the runtime your deliverable uses, in lowercase, with valid values of java/python/node/go; the version field is a semantic version number whose three parts x.y.z accept digits only (e.g., 1.0.0).
The root directory must contain the corresponding dependency declaration file according to your implementation: pom.xml for Java, requirements.txt for Python, package.json for Node.js, and go.modfor Go.
In addition to providing the dependency declaration file, all dependencies must be bundled inside the ZIP and adapted to the runtime environment; the runtime environment does not provide the ability to install dependencies over the network.
The environment variable DASHSCOPE_API_KEY is pre-configured and serves as the API Key for model invocation. It must be read from the environment variable; models cannot be invoked successfully by any other means.
The environment variables DASHSCOPE_BASE_URL and OPENAI_BASE_URL are pre-configured, serving respectively as the endpoint for invoking models via the DashScope API and via the OpenAI-compatible Chat API. DASHSCOPE_BASE_URL ends with /api/v1, and OPENAI_BASE_URL ends with /v1. They must be read from the environment variables; models cannot be invoked successfully by any other means.
The environment variable AGENT_LOG_DIR specifies the log output directory. Please write all relevant logs to `agent.log` under this directory.
Model usage scope: during the machine evaluation stage, only the models in the list below may be invoked; calls to any other model will fail.
Supported model list:

```
qwen-plus
cosyvoice-v3-flash
deepseek-v4-flash
deepseek-v4-flash-0731
deepseek-v4-pro
fun-asr
fun-asr-flash-2026-06-15
glm-5.1
glm-5.2
happyhorse-1.1-r2v
happyhorse-1.1-t2v
qwen-audio-3.0-tts-flash
qwen-image-3.0-pro
qwen-vl-max
qwen-vl-ocr
qwen3-asr-flash
qwen3-asr-flash-filetrans
qwen3-tts-flash
qwen3-vl-plus
qwen3.5-plus
qwen3.6-35b-a3b
qwen3.6-flash
qwen3.6-plus
qwen3.7-max
qwen3.7-plus
qwen3.8-max
wan2.7-i2v-2026-04-25
wan2.7-image
wan2.7-image-pro
wan2.7-videoedit
```

Model usage restrictions: MCP, agent applications, workflows, memory, knowledge-base retrieval, Embedding, and similar capabilities are not supported. Only the model inference capabilities required by this task are provided, invoked via the DashScope API or the OpenAI-compatible Chat API; invocation methods such as the Responses API are not supported. Built-in model tools are not supported. Uploading local files as model input is not supported — to pass images, audio, video, etc. as invocation parameters, you must use the result URLs returned by previous model calls, or correctly identify and use the URLs provided in the task data as input. The Agent must correctly handle any model rate-limiting responses it may encounter.
Artifact download: artifact URLs returned by models must be downloaded locally via HTTP and saved to the output directory specified in the input prompt as the final deliverables.
Network access restrictions: only the QwenCloud Platform model services and model artifact URLs are accessible; no other external network access is available.
Exit code convention: 0 indicates success, non-zero indicates failure.
Each run is limited to 30 minutes; exceeding this will cause the task to fail. Runtime memory usage must not exceed 4 GB.
The deliverable must support the --version argument: when launched with --version, it must output the same semantic version number as the version field in agent.json. For example, if agent.json is {"runtime":"python","version":"1.0.0"}, then python agent.py --version must output 1.0.0 and exit with code 0.
Example startup command: python agent.py --prompt "Please complete the Q&A task using the data in /data/dataset, and write the results to /workspace/output/result.json".
Technical documentation reference:

QwenCloud API Usage Guide - https://docs.qwencloud.com/developer-guides/getting-started/first-api-call.

2. Note B: Sample Prompt Used for Automated Testing
During the automated evaluation stage, the platform tests each Agent uploaded by participants by running it against the prompt below; this prompt stays consistent throughout the entire automated evaluation stage.

Sample Prompt

（The following is provided only as a reference for the system input Prompt used in automated evaluation. When debugging locally, replace the file paths and file names with the actual locations and names in your own environment.）

```
## Objective
 Read all information files for the target product in the `/home/user/ws/input/` directory, extract the required content, generate the output files according to the specifications, and save them to `/home/user/ws/output/`.

## Input

Input directory: `/home/user/ws/input/`
This directory contains several information files describing the target product. Their contents and relationships are as follows:

| File | Description | Notes |
|------|-------------|-------|
| product_info/product_basic.json | Basic listing data of the product, including title and SKU | `product_basic` is an illustrative file name only. The actual file name varies by product, so your Agent must discover the file under `product_info/` at runtime rather than hardcoding this name. |
| clothing_attributes.json | Apparel attribute definitions of the target platform | — |
| clothing_categories.json | Apparel category structure of the target platform | - |

## Output Requirements

1. Output directory: `/home/user/ws/output/`
2. For the given product, produce the following complete set of assets; each must follow the naming conventions and be field-parseable:

| Deliverable | Quantity | Format | Naming | Notes |
| --- | --- | --- | --- | --- |
| Product copy | 3 files | `.txt` / `.md` | `product_description_en`， `product_description_ko`， `product_description_pt` | One each in English / Korean / Portuguese, each copy file must include the product title; product information (the SKU and its component breakdown, and product attributes); the source platform name; the product ID and URL; each image name with its description; and the video name with its description.<br>Three language versions |
| Main image | 1 image | `.png` / `.jpeg` | `main_image` | Product main image |
| Detail images | 5 images | `.png` / `.jpeg` | `detail_image_1` – `detail_image_5` | Product detail images |
| Product video | 1 video | `.mp4` / `.mov` | `product_video` | Product introduction video |
| Strategy document | 1 file | `.txt` / `.md` | `strategy_document` | Describes the Agent's overall design approach and generation strategy |
```

3. The naming rules must be followed strictly — all outputs must use exactly the names above so the platform can locate and parse each asset type.

Note: The product copy comprises one file each in English, Korean, and Portuguese; the three files must be distinguished by language (e.g., `product_description_en` / `product_description_ko` / `product_description_pt`).

## III. Agent Deliverables Checklist
For each product, the Agent must produce the following complete set of assets:

Deliverable	Quantity	Format / Naming	Notes
Product copy	3 files	txt / md, named product_description	Three language versions: English, Korean, and Portuguese in different copy; Each product copy file must include the product title; product information, including the SKU, its component breakdown, and product attributes; the source platform name, product ID, and URL; each image name with its description; and the video name with its description.
Main image	1 image	named main_image	—
Product detail images	5 images	named detail_image_1 through detail_image_5	—
Product video	1 video	named product_video	—
Strategy document	1 file	txt/md, named strategy_document	Describes the Agent's overall design approach and generation strategy
All deliverables must strictly use the naming conventions above, so that the platform can locate and parse each asset type.


## IV. Evaluation Dimensions Reference
We have two stage evaluation : automated evaluation and expert review.

Automated evaluation stage: the China site — the Qianwen AI platform (www.qianwenai.com) — and the overseas site — QwenCloud (www.qwencloud.com) — run separate leaderboards, each presenting its public ranking. The top 30 entries from each leaderboard are then combined and advance to the expert review stage.
Expert review stage: the top 30 entries from the automated evaluation on both the China and international sites undergo a unified blind review and consolidated ranking by experts, who select the final Top 5 and determine the Gold, Silver, and Bronze award winners.
1. Automated Evaluation (Public Leaderboard)
Automated evaluation runs on each participant's most recent submission, keeps only the latest score and ranking, and updates the public leaderboard used for shortlisting. The evaluation dimensions are as follows:

Total automated-evaluation score = Σ (each dimension's score × its weight); ranking is based on the total score.

Dimension	Scope	Sub-dimension	Evaluation Method	Weight
A1 Content Compliance	Applies to all text assets, any recognizable text within images/videos, and visual elements within the imagery; judged solely against the platform's content-compliance rules for listing assets. No legal or intellectual-property judgment is made. Text and visual content are evaluated separately, and participants are responsible for researching AliExpress's listing rules.	Overall assessment	Compliant if the title, product detail copy, on-image/on-video overlay text, and visual elements contain no violations.	25%
A2 Asset Specification Compliance	Verifies the completeness and physical specifications of the Agent's output assets; does not evaluate how the ZIP archive is packaged or whether the content is correct.	Text content completeness; main image specification (jpeg/png, ≥800×800px, <1MB);
product detail image specification (jpeg/png, width and height both >260px, ≤5MB per image);
video (file exists and plays back normall, mp4/mov, <200MB)	Scored by sub-dimension; meeting the specification earns the score.	20%
A3 Category and Attribute Accuracy	Verifies category names, product attributes, and the enumerated values of sales attributes; this check only compares the participant's output against the reference answer. The algorithm does not independently determine the correct category.	Leaf category correctness;
product attribute match (each attribute key/value);
sales attribute (specification/variant value) match	Scored by sub-dimension; meeting the specification earns the score.	18%
A4 Localization Fit	Evaluates cultural and market differences in main and product detail images, copy tone and word choice, spelling, sizing and measurement units, cultural sensitivities, and related factors.	Image background standards;
wording appropriateness;
spelling and local phrasing;
sizing system;
units of measurement;
cultural sensitivities, including gender and body type;
adaptation to sales channels and use cases	Scored by sub-dimension; meeting the specification earns the score.	15%
A5 Product Factual Consistency	Text assets are checked for information sourcing; on-image/on-video overlay text is spot-checked only.	Sourcing matches the product source data;
each labeled attribute is consistent with the source data	Every verifiable attribute must cite its source and match the reference answer. Any attribute that is unsourced, inconsistent with the source data, or inconsistent with the reference answer is marked incorrect.	10%
A6 Image Usability Rate	All images generated in a single run.	Definition of "usable": meets the platform's image specifications and has no major quality defects.	To pass, at least the specified threshold (≥80%) of images from a single run must be "usable"; anything below the threshold fails.	7%
A7 Video Usability	All videos generated in a single run.	Definition of "usable": meets the platform's video specifications, plays back normally, and has no major quality defects.	"Usable" means the video plays back normally and has no unacceptable defects.	5%

2. Expert Review (Top 30 Blind Review)
The top 30 entries from each of the two leaderboards are combined and advance to the expert review stage, where industry experts conduct a unified blind review and consolidated ranking across the following dimensions:

Total expert-review score = Σ (each dimension's score × its weight).

Dimension	Description	Weight
Agent User Experience	Interaction flow, stability, and overall ease of use.	15%
Image Asset Quality	Usability and presentation quality of the main image and product detail images.	30%
Video Asset Quality	Usability and presentation quality of the product video.	20%
Overall Strategy	The soundness, completeness, and competitiveness of the localization and asset-generation strategy.	35%
Following expert review, a final Top 5 will be selected from the shortlisted entries, awarding Gold (1 winner), Silver (2 winners), and Bronze (2 winners).


## V. Troubleshooting Common Issues
If you run into issues while completing the task, please work through the following checklist first:

Was the API key used as required?
Do the output file names comply with the naming conventions?
Are all required assets complete?
Does the submitted code package meet the requirements?
Did the Agent's runtime exceed 30 minutes?
Is tIs the model being called one of the models in the required list?
The model call does not meet the requirements — check whether your input parameters comply with the task requirements and model restrictions.
DASHSCOPE_BASE_URL ends with /api/v1 and OPENAI_BASE_URL ends with /v1. When making calls, check whether the URL contains duplicate path segments or formatting errors.
Check whether the code package dependencies are complete.

These are common examples only and not an exhaustive list. If you encounter an issue not covered here, please review your Agent entry to diagnose it.