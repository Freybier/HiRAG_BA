# Evaluation metrics explained

What each column in `scores.csv` / `aggregate.json` means, how it is computed,
and how to read the values produced by `score.py`.

All scores are in the range $[0, 1]$ unless noted otherwise; higher is better.
The six `*_status` columns record whether a metric was `scored`, `skipped`
(preconditions missing, e.g. no reference) or `failed` (metric raised or
returned NaN). Only `scored` values enter the mean in `aggregate.json` — a `0.0`
is a real score and counts.

The first eight metrics come from RAGAS (the local checkout in `../ragas`) and
are computed by a judge LLM (the `evaluation.llm_model` deployment from
`config.yaml`) and/or embedding similarity. The last two are ID-based metrics
that compare source IDs directly, without an LLM.

## Answer-quality metrics

### `faithfulness`

**Question answered:** Is every claim in the generated answer supported by the
retrieved contexts, or did the system hallucinate?

**How it works:** The judge LLM breaks the generated answer down into atomic
statements, then verifies for each statement whether it can be inferred from the
retrieved contexts.

$$\text{faithfulness} = \frac{\text{statements supported by the context}}{\text{total statements in the answer}}$$

**Interpretation:** 1.0 = every claim is grounded in the retrieved context. This
measures hallucination _relative to the retrieved context_, not absolute truth —
a faithful answer can still be wrong if the context itself is wrong. Example
from the runs: q011 scored 0.25 because only part of the answer could be traced
back to the context.

### `answer_relevancy`

**Question answered:** Does the answer actually address the question (rather
than rambling, deflecting, or answering something else)?

**How it works:** The judge LLM generates several synthetic questions that the
given answer would be a good answer to, then measures the mean cosine similarity
(embeddings) between those synthetic questions and the original question.
Answers that are incomplete or redundant are penalized.

**Interpretation:** High scores (~0.95 in the runs) mean the answer is on-topic.
It says nothing about factual correctness — a confidently wrong but on-topic
answer still scores high.

### `answer_correctness`

**Question answered:** How close is the generated answer to the hand-written
`reference` answer?

**How it works:** A weighted combination of two components (defaults from the
local RAGAS checkout: weights `[0.75, 0.25]`):

1. **Factuality (weight 0.75):** the judge LLM extracts statements from the
   generated answer and the reference, classifies them as true positives
   (present in both), false positives (only in the answer) and false negatives
   (missing from the answer), and computes an F1 score ($\beta = 1.0$).
2. **Semantic similarity (weight 0.25):** embedding cosine similarity between
   the generated answer and the reference.

**Interpretation:** This is the metric that captures "did the system give the
right facts". It requires a `reference` in the dataset. Note the trade-off with
`answer_relevancy`: answers can be highly relevant yet only partially correct
(e.g. q010: relevancy 0.98, correctness 0.43).

### `factual_correctness`

**Question answered:** Which fraction of the factual claims agree between the
generated answer and the `reference` answer, in both directions?

**How it works:** The judge LLM decomposes the generated answer and the
reference into atomic claims, then verifies each claim via natural language
inference (NLI) against the other side: true positives (claims present in both),
false positives (claims only in the answer), false negatives (reference claims
missing from the answer). The score is the F1 over these sets ($\beta = 1.0$;
`mode="f1"` in `score.py`):

$$\text{factual correctness} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

**Interpretation:** Similar in spirit to the factuality component of
`answer_correctness`, but _pure_ claim-level F1 without the 0.25 semantic
similarity part, so it is stricter about missing reference facts and unsupported
extra claims. Run averages were low (0.24–0.35) because the hand-written
references are terse single facts while the HiRAG answers follow the mandated
5-section layout, so many reference claims and many answer claims have no
counterpart on the other side. Requires a `reference` in the dataset; the
retrieved contexts are not used.

## Retrieval-quality metrics (LLM-judged)

### `context_precision`

(RAGAS `ContextPrecision`; formerly `context_precision_with_reference`.)

**Question answered:** Of the retrieved contexts, are the _useful_ ones ranked
at the top — using the `reference` answer as ground truth for "useful"?

**How it works:** For each retrieved context item, the judge LLM decides whether
it was useful for producing the reference answer. The binary verdicts are turned
into an _average precision_ score, which rewards relevant items appearing early
in the ranking:

$$\text{AP} = \frac{\sum_k (P@k \times v_k)}{\text{number of relevant contexts}}$$

where $v_k \in \{0,1\}$ is the usefulness verdict at rank $k$.

**Interpretation:** This metric is rank-sensitive. HiRAG scores are low
(~0.1–0.3) by construction: a `hi`-mode query sends 100–200 context items
(entity descriptions, community reports, raw chunks) to the answer LLM, of which
only a handful are needed for a specific fact, so most verdicts are "useful:
no". Read it as "signal-to-noise ratio of the context", not as a retrieval
failure — see `context_recall` for the other side.

### `context_utilization`

(RAGAS `ContextUtilization`; formerly `context_precision_without_reference`.)

**Question answered:** Same as above, but each context item is judged against
the _generated answer_ instead of the reference answer.

**Interpretation:** Useful when no reference exists. It tends to track
`context_precision`; divergences indicate cases where the answer used different
context than the reference would have.

### `context_recall`

**Question answered:** Does the retrieved context contain everything needed to
produce the reference answer?

**How it works:** The judge LLM decomposes the `reference` answer into claims
and checks whether each claim can be attributed to the retrieved contexts.

$$\text{context recall} = \frac{\text{reference claims supported by the context}}{\text{total reference claims}}$$

**Interpretation:** ~1.0 in the runs means retrieval almost never misses
information the reference answer needs. Together with the low precision values
this paints the classic HiRAG picture: the context is _complete but noisy_ —
everything relevant is in there, buried under many marginally relevant items.

### `context_entity_recall`

**Question answered:** Are the _entities_ (named things: process names, document
titles, IDs, roles) mentioned in the `reference` answer present in the retrieved
contexts?

**How it works:** The judge LLM extracts the set of entities from the reference
answer ($GN$) and from the retrieved contexts ($CN$), then computes the set
overlap:

$$\text{context entity recall} = \frac{|CN \cap GN|}{|GN|}$$

**Interpretation:** A coarse, entity-level sibling of `context_recall` (which
checks full claims). It only asks "do the key names/IDs appear anywhere in the
context", not whether the context supports the statements about them — a
question can score ~1.0 here while `context_recall` is lower, and 0.0 here when
the context is about a different entity even if it is topically close. Values of
0.999… are RAGAS float artifacts of a perfect 1.0. Requires a `reference` and
retrieved contexts.

## Retrieval-quality metrics (ID-based, no LLM)

These two metrics ignore content entirely and compare the set of retrieved
ADONIS source IDs (`retrieved_context_ids`, resolved via
`source_id_mapping.json`) against the dataset's `reference_context_ids`. They
are only computed for runs with `id_granularity: adonis_source_record`
(currently HiRAG runs only) and need no judge LLM.

### `id_context_precision`

$$\text{ID precision} = \frac{|\text{retrieved IDs} \cap \text{reference IDs}|}{|\text{retrieved IDs}|}$$

**Interpretation:** Which fraction of the source records that contributed to the
context were actually the records the question is about. Low values (e.g. 0.11 =
1 of 9 retrieved source records is a reference record) again reflect the wide,
graph-driven context of HiRAG.

### `id_context_recall`

$$\text{ID recall} = \frac{|\text{retrieved IDs} \cap \text{reference IDs}|}{|\text{reference IDs}|}$$

**Interpretation:** Were all reference source records retrieved? 1.0 = every
record the question was built from appears in the context's provenance. q034
scoring 0.5 means only one of its two reference records was retrieved — matching
its `context_recall` of 0.6, the only question where content was genuinely
missing.

## How the metrics fit together

| Metric                                      | Needs                            | Measures                       | Failure it detects        |
| ------------------------------------------- | -------------------------------- | ------------------------------ | ------------------------- |
| `faithfulness`                              | answer + contexts                | grounding of answer in context | hallucination             |
| `answer_relevancy`                          | question + answer                | on-topicness                   | deflection / rambling     |
| `answer_correctness`                        | answer + reference               | factual overlap with truth     | wrong/incomplete answer   |
| `factual_correctness`                       | answer + reference               | claim-level F1 vs. truth       | missing/invented facts    |
| `context_precision` / `context_utilization` | contexts (+ reference or answer) | ranking signal-to-noise        | noisy context             |
| `context_recall`                            | contexts + reference             | completeness of context        | missing information       |
| `context_entity_recall`                     | contexts + reference             | entity presence in context     | missed key entities       |
| `id_context_precision`                      | retrieved + reference IDs        | source-level noise             | irrelevant source records |
| `id_context_recall`                         | retrieved + reference IDs        | source-level completeness      | missed source records     |

Typical diagnosis pattern:

- Low `context_recall` / `id_context_recall` → the retrieval step lost
  information; fix indexing/retrieval.
- High recall but low `faithfulness` → the answer LLM is adding unsupported
  claims; fix prompting or the answer model.
- High recall, high faithfulness, low `answer_correctness` /
  `factual_correctness` → the answer is grounded but misses or distorts the
  specific facts of the reference; check question phrasing and reference
  granularity.
- High `context_recall` but low `context_entity_recall` → the claims are
  supportable but the exact entities of the reference are absent; check whether
  the question's target record was retrieved at all.
- Low `context_precision` with high recall → expected for HiRAG's wide contexts;
  only actionable if it starts hurting faithfulness or cost.

## Caveats

- **Judge bias:** the judge LLM comes from the same model family as the answer
  models (see README "Bias warning"), so self-preference can inflate scores.
  Compare systems _relative_ to each other, not as absolute quality.
- **LLM-judged metrics are non-deterministic:** re-scoring the same run yields
  slightly different values (visible in repeated runs, e.g. answer_correctness
  of q001 varying between 0.55 and 0.64).
- **ID metrics are set-based:** duplicates and ranking do not matter, and they
  only see source-record provenance, not which chunk inside a record was used.
- **`n_contexts`** (the column after `record_status`) is the number of context
  items the system sent to the answer LLM for that question — the denominator
  context for the precision metrics.
