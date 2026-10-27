# Design rules for this project

Supplements `~/.claude/CLAUDE.md`. These are rules, not context — each one exists because a
claim here had to be retracted.

## Attribution: ours versus theirs

- Before attributing a defect to third-party code, grep their repository for the thing itself.
  If it is not there, the bug is ours. Notes on our own fork document *our* fixes; they are not
  evidence about their code.
- Write "our port" or "their implementation", never "the implementation".
- When an experiment imports their code, say so and cite the loader line. When it uses our
  reimplementation of their method, say that instead — it is a weaker claim.

## Claims

- Every claim carries its source: a file and line, a line in the paper, or a result file. A
  claim you cannot point at is a hypothesis, and must be labelled as one.
- Every claim carries its confidence. "One unreproduced run" is a confidence level; state it.
- Separate measured from inferred. Mechanism stories are usually inferred.
- Keep retractions visible. When a claim in a document changes, add a row to that document's
  corrections table instead of editing quietly.
- Grep hits are not a coverage check. Before concluding that content exists elsewhere, compare
  at section level — keyword matches find the results of a derivation, not the derivation.
- Name the configuration a number came from. A result in one setting is not evidence about
  another setting that shares a name.

## Experiment design

- One arm changes one thing. If an arm must change two, add the intermediate arm, or the effect
  will be attributed wholly to the wrong cause.
- Pair the contrast when arms share a target. Marginal spread across seeds and conditions is
  dominated by between-condition variation and hides the comparison.
- Report effect size against available headroom, not only significance. A result can be clear at
  three standard errors and still be worth a tiny fraction of the distance to the ceiling.
- Check whether a criticism of their setup also applies to our harness.
- Shake-out numbers do not count. Nothing is promoted into the library on a smoke test.
- When you try to break your own finding, keep the attempt in the script as a permanent arm.
- Check the load before launching locally; prefer CPU over queueing behind another process.

## Criticising external work

- Reproduce one of their positive results first. Nothing else is credible without it.
- Use their own ablation when they have one. It is a better control than anything we would
  build, and it holds the rest of their pipeline fixed.
- Check a parameter's initialisation before claiming it degenerated. "It collapsed" means
  nothing until you know it did not start collapsed.
- Ask the discriminating question before calling something broken: **does a better parameter
  setting exist inside the same family?** Yes, and the fit misses it, is an *optimisation*
  failure. No, after a real search, is a *representational* failure. They need different
  evidence and imply different fixes; conflating them produces unreadable analysis.
- Prefer a containment argument to an analogy. Showing what a family can and cannot express
  yields a falsifiable prediction. Analogies belong underneath an argument, never instead of one.
- Scope a criticism to the configuration it was measured in, and scope a constructive result the
  same way. A better fit by one metric can make the target application worse.

## Writing documents

- Documents live in git. Publish a page only as a shareable snapshot of something committed.
- Write for the authors of the work being discussed. Keep the field's vocabulary; make the
  structure plain, not the words.
- Order each section: the problem, why it is a problem, the evidence, how solid it is, how to
  re-run it. Asserting a defect is not explaining one.
- Give every document a section on what the work does not establish.
- Treat another agent's results as evidence to check, not as input to accept. Verify against
  something independent before building on them.
