# Together low-cost evaluation lane

Koschei Sentinel uses Together only as an evaluation provider. The signed deterministic verdict remains final, and every Together output must pass the same authority, grounding, abstention, privacy, and confidence gates as every other candidate.

## Billing reality

Together serverless inference is usage billed. Account credits or promotions may cover early runs, but the repository never assumes a model is permanently free. Pricing metadata in the candidate registry is a dated safety snapshot and must be reviewed before a live run.

The initial registry contains two structured-output candidates:

- `openai/gpt-oss-20b`
- `Qwen/Qwen3.5-9B`

Each candidate is limited to four requests, 512 output tokens per request, and a conservative maximum estimated cost of `$0.01` for one comparison run.

## Plan without network access

```bash
sentinel-compare \
  --suite fixtures/evals/suite.safe.jsonl \
  --registry fixtures/models/candidates.together.low-cost.json \
  --plan-only
```

The planner treats each UTF-8 input byte as one token, which deliberately overestimates normal JSON prompt tokenization. It then adds the configured maximum output tokens. A candidate is rejected before credentials or DNS are touched when its request count or estimated maximum cost exceeds policy.

Select one candidate while planning:

```bash
sentinel-compare \
  --suite fixtures/evals/suite.safe.jsonl \
  --registry fixtures/models/candidates.together.low-cost.json \
  --candidate sentinel-together-gpt-oss-20b \
  --plan-only
```

## Run locally

Store the key only in the environment:

```bash
export TOGETHER_API_KEY='replace-locally'
```

Then run exactly one candidate:

```bash
sentinel-compare \
  --suite fixtures/evals/suite.safe.jsonl \
  --registry fixtures/models/candidates.together.low-cost.json \
  --candidate sentinel-together-gpt-oss-20b \
  --output-dir build/comparisons/together-gpt-oss-20b \
  --allow-network \
  --require-all
```

Together candidates cannot configure a custom base URL. The adapter always uses `https://api.together.ai/v1`, requires `TOGETHER_API_KEY`, and sends chat-completions requests with JSON-object output enabled.

## Run from GitHub Actions

Add `TOGETHER_API_KEY` as a repository Actions secret. Never paste it into an issue, pull request, workflow input, candidate file, or chat message.

Open the `together-benchmark` workflow, choose one candidate, explicitly confirm the possible spend, and run it manually. Normal pull-request and push CI never receives the secret and never calls Together.

The workflow first prints the conservative cost ceiling, then runs the selected candidate, and retains the comparison artifact for seven days.

## Updating prices or models

Before changing a pricing snapshot:

1. Verify the model still exists in Together's serverless catalog.
2. Update input and output prices plus `pricing_as_of`.
3. Run `--plan-only`.
4. Keep `max_estimated_cost_usd` at or below the approved limit.
5. Run the full secrets-free CI before any live benchmark.
