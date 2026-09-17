# LitJev

**An implementation of our hypothesis about Jev.** Built with Hugging Face
Transformers, LitJev implements a JEV-like decision layer that adapts off-the-shelf
LLMs into typed decision APIs, with a browser playground and HTTP server included.

这是我们对 Jev 实现方式的猜想实现：基于 Hugging Face Transformers 构建
JEV-like 决策层，将 off-the-shelf 大模型快速接入 JEV-like API。
这一猜想来自公开信息，不代表 Jev 的真实内部架构；当前首先保证 Qwen 可用。

Evaluate Choice, Score and Noul questions in one request, including ten-question batches. LitJev reads
candidate scores from the model's output head and builds typed responses in Python:
no generated JSON, no answer-text parsing, and no generated answer tokens.

> Independent research project, not affiliated with or endorsed by TypeSafe AI.
> Not the official Jev implementation. Request/response JSON follows the public
> Jev schema; model behavior, confidence values and hosting features are not identical.
> First supported target: `Qwen/Qwen3.8-27B`. Probabilities are **not calibrated by
> default**. Other checkpoints are not guaranteed to work.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), obtain this
repository, and run from its root:

```bash
git clone https://github.com/zhengxuyu/litjev.git
cd litjev
uv run --locked litjev --model Qwen/Qwen3.8-27B
```

Open **http://127.0.0.1:8000/**. This command installs dependencies and starts both
the playground and API. Load the example, then submit. The first inference request
downloads/loads the model and can take several minutes. Later requests reuse it.
Opening the page does not load weights.

For an existing checkpoint:

```bash
uv run --locked litjev --model /path/to/qwen-checkpoint --device-map cuda:0
```

**Not yet published to PyPI.** The commands above run this checkout. Do not assume
`uvx litjev` or `pip install litjev` installs this project before a release.

### Requirements

- Python 3.11+ and uv.
- Tested hardware: one NVIDIA H100 80 GB, BF16. Budget roughly 54 GB for 27B BF16
  weights alone, plus runtime overhead and branch caches. An 80 GB GPU is the tested
  starting point, not a general minimum.
- Linux uses the locked PyTorch CUDA 12.8 distribution and needs a compatible driver.
  Model downloads need additional disk space and network access.
- `--device-map auto` is the default. Multi-GPU/offload configurations are not
  benchmarked here. CPU-only 27B inference is not a supported performance target.

Options: `--model`, `--revision`, `--dtype` (`bfloat16`, `float16`, `float32`),
`--device-map`, `--port`, `--calibration`. Run `uv run litjev --help` for usage.
The server binds to loopback only, with one process owning the model.

## HTTP API

Interactive API documentation: **http://127.0.0.1:8000/docs**.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

Example request:

```json
{
  "model": "litjev",
  "state": "Answer each question using its listed options.",
  "questions": {
    "math": {
      "type": "choice",
      "instructions": "What is 2 + 3?",
      "criteria": {"A": "4", "B": "5", "C": "6"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the state?",
      "criteria": ["Not urgent", "Urgent", "Critical"]
    },
    "needs_review": {
      "type": "noul",
      "instructions": "Does this need human review?"
    }
  }
}
```

This is a **Jev questions mapping, not JSON Schema**. The bundled request and
playground example contain the same ten questions. `state` accepts a string, object,
or array. `instructions` and criterion descriptions accept strings, objects, arrays,
or null. `instructions` may be omitted. Question IDs are never sent to the model.

| Type | Criteria | Answer fields |
| --- | --- | --- |
| `choice` | Map of up to 255 option keys to descriptions | `type`, `choice`, `probabilities`, `confidence` |
| `score` | Ordered array of 2–10 level descriptions | `type`, `score`, `legend`, `probabilities`, `confidence` |
| `noul` | Optional map with `true` and/or `false` descriptions | `type`, `noul` |

Choice returns the original option key, even when it contains multiple tokens.
Score returns the probability-weighted level index (0-based), not the winning level.
Noul returns P(yes), not a boolean. Score `legend` preserves original descriptions.

The standard response contains **only** `model`, `answers`, and
`usage: {input_tokens, output_tokens}`. `output_tokens` is zero.
Request `model` must be `litjev` (local alias) or the loaded checkpoint ID; the
response reports the actual checkpoint, never pretends to be `jev-latest`.

For timings and raw logits, POST the same body to **`/v1/systemone/debug`**.
Its envelope is `{result, diagnostics}`: `result` is the standard response;
`diagnostics` holds `fields.*.{logits, probabilities, max_probability, provenance}`,
`forward_calls`, `confidence_method`, `calibration_fitted`, and `timing`.
The playground and visual games use this extension. `GET /health` stays lightweight.
The old `/v1/calibrated-schema` and `/v1/batch-mcq` routes have been removed;
old `schema`/`enum`/`boolean` request objects are rejected rather than silently converted.

The playground displays probability bars, JSON, logit provenance, and timing.
`model_setup_seconds` includes first-use loading; `decision_seconds` includes
tokenization, inference-lock waiting, and inference. `total_seconds` sums these
server phases. Browser round-trip time also includes transport. These are not
GPU-kernel-only timings.

## Visual games: Doom, chess, and film

Play from screenshots with the same off-the-shelf Qwen weights—no training. The
ported Doom (seven buttons) and chess (five controller keys) examples share a
Gymnasium interface and a LitJev policy supporting either local inference or HTTP.

```bash
# Start the model server, then run a game in another terminal.
uv run --locked --extra games litjev --model Qwen/Qwen3.8-27B
uv run --locked --extra games litjev-play chess --max-steps 100 --output runs/chess.json
uv run --locked --extra games litjev-play doom --max-steps 100 --output runs/doom.json
uv run --locked --extra games litjev-film runs/chess.json --output runs/chess.html
```

Open `/film` to inspect traces, or open the standalone HTML. The screenshot playground
also accepts PNG/JPEG uploads. Only the debug API adds an optional base64 `image` field;
it never substitutes hidden game state or text descriptions for pixels.
See [visual games](docs/visual-games.md) for the complete interface, timing boundaries,
MP4 export, and limitations. No gameplay quality or real-time performance is promised.
The examples are adapted from [jevlike](https://github.com/vinnylarouge/jevlike), with
[MIT attribution retained](THIRD_PARTY_NOTICES.md).

## Python usage

```python
from dataclasses import asdict

from litjev.backend import ModelSettings, TransformersScorer
from litjev.decision import SchemaDecisionEngine
from litjev.schema import DecisionSchema

model = "Qwen/Qwen3.8-27B"
scorer = TransformersScorer.load(ModelSettings(model_id=model))
engine = SchemaDecisionEngine(scorer, model_id=model)
schema = DecisionSchema.from_mapping({
    "math": {
        "type": "choice",
        "instructions": "What is 2 + 3?",
        "criteria": {"A": "4", "B": "5"},
    }
})
result = engine.decide("Choose the correct answer.", schema)
print(result.answers["math"].choice)
print(asdict(result))
# For logits and provenance: engine.evaluate(state, schema) -> Evaluation(result, diagnostics)
```

## How it works

1. **Shared prefill:** encode only the state and generic instructions once.
2. **Cached branches:** replicate KV/recurrent states and batch each question's
   instructions and criteria in a second forward pass, ending with `Answer:`.
   Original option keys map to internal letter codes (A–Z, then eligible AA…ZZ/AAA…ZZZ), verified as single tokens
   at this boundary. Unsupported tokenizers are rejected, never truncated.
3. **Readout:** take `output.logits[i, len(suffix_ids[i]) - 1, candidate_ids[i]]`.
   For Qwen, these are vocabulary-head (`lm_head`) logits after the final decoder
   normalization, at the final input position (the colon for Qwen), predicting
   a space-prefixed internal code token. These codes are not generated.
4. **Typed response:** normalize candidate logits, select the maximum, and build
   JSON in code. No autoregressive answer generation is performed.

Branches cannot see one another's questions or results. Renaming a question ID
does not change its model input. This schema migration changes the prompt compared
with the original catalog-based version, so historical accuracy, calibration profiles
and latency numbers must not be applied to it without re-evaluation.

### Relationship to Jev

LitJev is inspired by typed decision APIs, but does not reproduce Jev's training
or proprietary internals. In this release:

- `/v1/systemone` follows the public `model / state / questions` request and typed
  response shapes, including structured criteria. See the [migration matrix](docs/jev-schema.md).
- Jev's exact confidence formula is not published in the referenced docs. LitJev
  uses normalized Gini concentration: `(K * sum(p_i²) - 1) / (K - 1)` for K > 1,
  and 1 for a single option. Uniform distributions yield 0, point masses yield 1.
  This is **not max probability**, not an accuracy estimate, and not a claim of
  numerical parity with Jev.
- No RLCD training, learned correctness head, or one-forward guarantee is provided.

References: [TypeSafe documentation](https://docs.typesafe.ai/introduction) and the
[RLCD reference implementation](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD).
These are independent external projects, not endorsements.

## Calibration

Default debug diagnostics have `calibration_fitted=false`. Normalization does not establish
calibration. Optional post-hoc temperature fitting is available:

```bash
# With the uncalibrated server running:
uv run litjev-mmlu --split validation --limit 70 --export-logits validation-logits.jsonl
uv run litjev-calibrate validation-logits.jsonl --model Qwen/Qwen3.8-27B --output calibration.json
# Stop the old server, then restart with the profile:
uv run litjev --model Qwen/Qwen3.8-27B --calibration calibration.json
```

Never fit on test labels. Keep model revision, prompt, option order, and evaluation
protocol fixed. A fitted temperature does not guarantee calibration on another
population or safe out-of-distribution routing. Do not use this preview as the sole
authority for consequential actions.

## Evaluation

With the server running:

```bash
uv run litjev-mmlu --split test --limit 10 --output mmlu-results.json
```

Downloads `TIGER-Lab/MMLU-Pro`, sends ten questions per request, and scores labels
locally. Gold answers and CoT explanations never enter inference prompts. This is
**direct-answer scoring, not the standard CoT benchmark protocol**. `--prepare-only`
exports requests without inference; `--revision COMMIT` pins the dataset snapshot.
Incomplete final batches use duplicate padding excluded from metrics. Invalid or
non-MCQ records are skipped with reasons.

A **historical, pre-migration** ten-question smoke experiment on one H100 80 GB achieved 9/10 and approximately
0.472 s per warm ten-question request (mean of three repeats). This is not a full
MMLU-Pro score or a latency guarantee. Repeated questions are not extra test examples;
loading, queueing, and HTTP overhead are excluded from this measurement. It used the
old shared question catalog, not the current isolated question
branches. No current 27B accuracy/latency result is claimed by this migration.

See [Slurm usage](docs/slurm.md) for configurable cluster launchers. Raw local
experiment outputs are excluded from the public distribution because they contain
machine paths, hostnames, and dataset text.

## Development

```bash
uv sync --locked --extra dev --extra games
uv run pytest -q
uv run ruff check .
uv build
```

Tests use fake engines and tiny randomly initialized Transformers models, not 27B
weights. They verify two-forward cache scoring, schemas, API behavior, and calibration
utilities. Browser assets are bundled in the wheel.

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[release checklist](docs/releasing.md). No authentication, request quota, or public
hosting service is included. Keep the server local or use an authenticated reverse
proxy before granting remote access.

## License

[Apache License 2.0](LICENSE) for this repository's original code. Adapted jevlike
example files retain their [MIT license](THIRD_PARTY_LICENSES/jevlike-MIT.txt);
see [third-party notices](THIRD_PARTY_NOTICES.md). Model weights,
datasets, and third-party dependencies retain their own licenses and are not bundled
here.

When redistributing LitJev or derivative works, comply with Apache-2.0 Section 4:
provide a copy of the license, mark modified files, retain applicable source notices,
and reproduce applicable attribution from [NOTICE](NOTICE) in a permitted location.
These requirements concern redistribution; they do not require every private use or
hosted API response to display a citation. The full license controls.

## Citation

If you use LitJev in research, benchmarks, publications, or a project, please cite
it and acknowledge its contribution using the entry below. Please carry this
attribution into your project's documentation or acknowledgments. This citation
request is not an additional condition of the Apache-2.0 license; redistribution
must still preserve the applicable notices described above.

如果你在研究、评测、论文或项目中使用 LitJev，请携带以下引用，在文档或致谢中
注明来源。再分发时必须按 Apache-2.0 保留适用的许可证和 NOTICE 归属声明；
学术引用是项目请求，不是对 Apache-2.0 另加的限制。

```bibtex
@software{litjev2026,
  author  = {{ZhengxuYu}},
  title   = {LitJev: A JEV-like Decision Layer for Off-the-Shelf LLMs},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/zhengxuyu/litjev}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).
