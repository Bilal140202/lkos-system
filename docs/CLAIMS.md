# Claims & Contradictions

## Extraction (deterministic, labeled heuristic)

Families: numeric metrics ("Acme Corp revenue was $10M in 2024"), release
verbs ("launched/acquired/announced/..."), copula ("X is Y"). Sentence
offsets recorded (start/end in chunk). Negation lowers confidence and feeds
the taxonomy. Validity bounds parse since/from/as of/in -> valid_from and
until/through/by -> valid_until.

## Unit normalization

"$10 million" = "$10M" = "10,000,000" = 10_000_000; commas/%/B/K handled.
(left-to-right scan; v0.1 kept only the last whitespace token.)

## Conflict taxonomy

| kind | meaning |
|---|---|
| same-period-disagreement | same metric, same period, >5% delta |
| cross-period | same metric, different periods (context, not error) |
| undated-disagreement | no periods parseable |
| negation-conflict | assert vs negate same predicate |

Cost bound: compared against the most recent 40 same-metric claims (indexed
(subject_key, predicate_key) lookup); at most 4 conflict rows materialized
per claim — disagreement existence is the signal; full pairs are quadratic
and uninformative (bench-stall measurement in whitepaper section 6).

Disagreement is preserved with both sources and explanations — never merged.
