# Decisions

## What I deliberately did not do, and why

I deliberately left production hardening out of the provided `fetch_url` live path. Replay mode never issues the request, and a correct live-mode fix requires more than a URL prefix check: it must enforce private-address and DNS/redirect boundaries plus response resource limits. The current path would be unacceptable on a customer network because a model-controlled URL could reach internal resources or consume excessive memory and bandwidth. I would block enabling live tools outside an isolated sandbox until those controls and adversarial tests were in place.

## How I would know this works

I would keep the current 34-test regression suite and deterministic replay as release gates for control-flow invariants such as approval order, caller-owned plan integrity, revision limits, and audit chronology, with zero tolerance for any pre-approval tool execution. For live quality, I would run a versioned set of 30 representative requests three times per candidate release and have coordinators score constraint satisfaction, source grounding, replan appropriateness, and whether the draft needs a major rewrite. During a pilot I would track the rate of unusable or error runs, acceptance without a major rewrite, and median time to an accepted draft against the roughly 40-minute manual baseline in the brief. I would pause rollout if unusable runs exceeded 5%, and reconsider product value if fewer than 80% of drafts avoided a major rewrite or median accepted-draft time improved by less than 50%. Those are initial pilot gates to calibrate with the customer, and a passing replay suite would never be treated as evidence of live factual quality.

---

**AI assistance:** An AI coding assistant helped inspect the codebase, draft changes, and challenge edge cases; I reviewed the decisions and validated them against the repository contracts, tests, and replay traces.

**Time spent:** About 3.5 hours, primarily on repository and contract analysis, implementation, regression tests, and replay/UI verification.
