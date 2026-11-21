# P2 preparation

Prepared split 0 and 30 additional 50/50 video splits per target. Split 0 exactly
retains the original manifest roles and must retain the original model choices.
New HERBench splits are balanced within source collection, with floor/ceiling
allocation randomized for odd-sized collections and exactly half the videos in
the pilot. Perception splits select half the videos. All questions inherit their
video assignment. No labels were read during split generation. No duplicate
pilot subsets or video leakage were found.

New split seed = target base seed + 1000003 * split_id. Base seeds are 20260802
and 20260803. These are new predetermined split seeds, not modifications of an
archived run. No fits, retuning, bootstrap evaluations, or gain distributions have
run. Execution needs the missing target probability archives (or verified full
features) and the compatible estimator environment. The canonical metric is
../P0/canonical_metric.py. Current output is preparation only.

No performance table or LaTeX result snippet is available before execution.
