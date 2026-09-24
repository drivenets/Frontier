import sys, re
plan = open("/home/dn/amd-playground/.claude/plans/linear-op-recollection-contract-a.md").read()
def section(name):
    m = re.search(rf"\n## {re.escape(name)}\n(.*?)(?=\n## |\Z)", plan, re.S); return m.group(1).strip() if m else "(missing)"
req = section("Requirements") + "\n\n### Non-Goals\n" + section("Non-Goals") + "\n\n### Constraints\n" + section("Constraints") + "\n\n### Success Criteria\n" + section("Success Criteria") + "\n\n### Explicit User Decisions\n" + section("Explicit User Decisions")
src = section("Reliable Sources")
out = f"""You are an independent plan reviewer (ForgeLoop plan-loop Stage 6b). REVIEW ONLY: do not modify, create or delete any file. The repository under plan is /home/dn/Frontier-qwen3-profiling (branch smatar/qwen3-30b-mi355-profiling, HEAD ec63f31); you may open any file there to verify the plan's citations and feasibility.

Review this plan against the stated requirements and source authority map.
Return exactly:

OVERALL: PASS | FAIL
BLOCKING_FINDINGS:
  - <severity>: <section> — <issue> — <required fix>
NON_BLOCKING_FINDINGS:
  - <severity>: <section> — <issue>
CHECKLIST_RESULTS:
  - REQ coverage: PASS | FAIL — <reason>
  - Source traceability: PASS | FAIL — <reason>
  - Unresolved decisions: PASS | FAIL — <reason>
  - Placeholder scan: PASS | FAIL — <reason>
  - Handoff completeness: PASS | FAIL — <reason>
  - Feasibility against the repo: PASS | FAIL — <reason>
FIX_BRIEF:
  - <exact plan section change, or "none">

=== REQUIREMENTS ===
{req}

=== SOURCE MAP ===
{src}

=== PLAN ===
{plan}
"""
path = sys.argv[1]; open(path, "w").write(out); print(path, len(out), "chars")
