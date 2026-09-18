# System Two: self-routed slow thinking

LitJev's standard endpoint is pure System One: one prefill, one branch forward, zero
generated tokens. `/v1/systemtwo` adds a slow path that the model routes to itself.
The backbone is never modified or trained; the only trained component is a small
**decision head** read beside `lm_head` at the answer boundary. Design notes are in
[`design/self-routing-architecture.md`](design/self-routing-architecture.md).

## How a request flows

1. Fast path as usual. In the same branch forward, the readout hidden states at
   `Answer:` are captured for the layers the head was trained on.
2. The head predicts, per question, a four-class outcome distribution:
   P(fast right, slow right), P(fast right, slow wrong), P(fast wrong, slow right),
   P(fast wrong, slow wrong). `confidence` becomes P(fast right), a calibrated value
   rather than the concentration statistic.
3. Routing: escalate a question when
   `P(fast wrong, slow right) - P(fast right, slow wrong) > lambda`. `lambda` is a
   request-time knob; it never enters training.
4. Slow path for escalated questions only: the branch is re-run with the backbone's own
   `<think>…</think>` generation (up to `budget` tokens), then the **same** `Answer:`
   boundary is read again. Answers stay typed; nothing is parsed from text.

`usage.output_tokens` counts thinking tokens. The standard response shape is unchanged,
so a client can keep the Jev schema and switch endpoints.

```json
{
  "model": "litjev",
  "state": "...",
  "questions": {"...": {}},
  "routing": {"lambda": 0.05, "budget": 512}
}
```

Omit `routing` to use the server default (`--lambda`, `--think-budget`; a budget of 0
disables routing). `/v1/systemtwo/debug` adds `fields.*.outcome`, `escalation_gain`,
`system` (`one` or `two`), the thinking text, and `diagnostics.routing`.
Image states are not supported on the slow path yet.

## Training the head

Labels come from the backbone itself: run both paths, compare each with the gold
answer, and record which of the four outcomes happened. No human labels of "should it
think" are needed. Gold labels never enter prompts.

```bash
# 1. Collect records (features + outcomes). Layers are hidden_states indices;
#    -1 is the final normalized output, 40 is decoder layer 40's output.
uv run litjev-collect --model Qwen/Qwen3.8-27B --split test --limit 2000 \
  --layers -1,40,48 --budget 512 --output records-0.npz

# 2. Layer sweep, train, evaluate on held-out categories, save the head.
uv run litjev-train-head records-*.npz --output decision-head.safetensors \
  --report decision-head-report.json --holdout-fraction 0.25 --select-layers 1

# 3. Serve with routing.
uv run litjev --model Qwen/Qwen3.8-27B --decision-head decision-head.safetensors \
  --lambda 0.05 --think-budget 512
```

The report lists per-layer AUROC for predicting fast-answer correctness, the chosen
layers, baselines (max probability and concentration), the head's AUROC, and a
coverage–accuracy curve over `lambda`. The split is by MMLU-Pro category so numbers
reflect transfer to unseen subjects. The head file records the model ID, revision,
prompt format and layers; serving refuses a head that does not match.

On Slurm, `scripts/collect.sbatch` shards collection across an array:

```bash
export LITJEV_MODEL=/path/to/checkpoint LITJEV_LAYERS=-1,40,48 LITJEV_SHARD=500
sbatch --array=0-9 --partition=P --account=A scripts/collect.sbatch
uv run litjev-train-head results/records-*.npz
```

## What to expect and what is not claimed

- Thinking is autoregressive. A 512-token budget on 27B under Transformers is tens of
  seconds per escalated batch; `/v1/systemtwo` is not a 500 ms product.
- The slow path is the backbone's native thinking; its quality is whatever the
  checkpoint can do. The head only decides when it is worth paying for.
- The head is a probe. If its AUROC is not clearly above the concentration baseline,
  change layers or data coverage before touching anything else.
- MMLU-Pro is choice-only. `score` and `noul` routing needs labelled data of those
  types; the head carries a question-type feature but has not been trained on them.
