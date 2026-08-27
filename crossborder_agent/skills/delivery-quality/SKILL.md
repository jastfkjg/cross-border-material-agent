---
name: delivery-quality
description: Plan and evaluate a bounded A1-A7 delivery, prioritizing task-weighted correctness and non-destructive repairs.
---

# Delivery Quality

## Compilable rules

- [manager][soft] Allocate the bounded repair budget by expected rubric-weighted gain, protecting A1, A2, A3 and A5 correctness before lower-impact polish.
- [manager,final-review][hard] Never interpret reviewer failure or uncertainty as artifact failure and never replace an accepted asset unless a new candidate passes hard gates and is demonstrably better.
- [manager,final-review][soft] Prefer a targeted field or single-asset repair with the highest expected weighted gain over broad regeneration.
- [final-review][hard] Distinguish physical contract validity from semantic readiness and score A1-A7 from delivered artifacts, not producer explanations.
- [final-review][hard] Require no blocker and no major A1, A2 or A5 defect before declaring delivery ready; report incomplete semantic stages explicitly.
