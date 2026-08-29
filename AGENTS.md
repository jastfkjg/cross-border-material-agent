# Repository guidance

## Testing autonomous agents

- Keep the automated test suite small, deterministic, and focused on whether the agent can run successfully and return structurally valid output.
- For LLM-directed or autonomous behavior, assert externally observable outcomes and hard safety/data-integrity contracts. Do not assert exact reasoning, prompt wording, search terms, tool-call order, implementation-specific intermediate state, or exact call/turn counts unless that timing or budget is itself an explicitly documented product contract.
- Preserve at least one lightweight end-to-end smoke test for every supported agent protocol. A smoke test should prove that a complete execution can finish and produce a valid result; it must not prescribe the semantic exploration strategy.
- Test deterministic tools only at their stable public boundary: accepted inputs, result shape, validation, grounding, error handling, and resource limits. Prefer small synthetic fixtures or dynamically discovered fixture relationships over hard-coded benchmark product, category, attribute, and value IDs.
- Measure semantic quality, model autonomy, and generalization in separate evals using unseen or holdout examples and aggregate metrics. Do not turn the current sample answers into unit-test routing rules or require one exact output when multiple valid outputs or trajectories exist.
- Before adding a regression test, confirm that it protects a documented behavior or a strategy-independent failure mode. Do not add tests merely to freeze the current implementation, current model behavior, or one observed execution trace.
- When an implementation is optimized, tests should continue to pass if the public outcome and safety contracts are unchanged, even when prompts, models, queries, tool usage, batching, number of steps, or internal architecture change.

## Generalization and sample data

- Treat the eleven currently supplied products as examples for understanding the input schema and validating the basic workflow, not as a complete representation of evaluation or production inputs. Assume unseen products, categories, attributes, values, languages, and combinations will be used.
- Minimize hard-coded semantic rules and product-specific heuristics. Production behavior must not branch on sample offer IDs, filenames, expected category/attribute/value IDs, sample titles, or hand-picked phrases derived from the eleven examples.
- Do not encode the sample answers indirectly through alias tables, fallback mappings, keyword-to-ID tables, exception lists, prompt examples, ordering assumptions, or tests that steer the agent toward known outputs.
- Prefer general mechanisms driven by the supplied data and runtime evidence: schema discovery, bounded generic queries, explicit relationships, model-directed exploration, structural validation, and provenance/grounding checks.
- Add a deterministic rule only when it expresses a genuinely domain-wide invariant, protocol requirement, safety constraint, or data-format contract. Document the invariant and ensure the rule remains valid for unseen products; do not use deterministic rules to replace an open-ended semantic decision merely because they improve the current samples.
- Evaluate changes for generalization risk. A change that improves the eleven examples but depends on their wording, IDs, category mix, ordering, or known answers should be treated as overfitting and must not be used as production selection logic.
