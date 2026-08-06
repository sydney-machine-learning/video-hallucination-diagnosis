# Current TRACE-RC results

The two confirmation directories are immutable copies of the downloaded Katana
analysis bundles.  Each includes an `archive.sha256` manifest.  The
`posthoc_robustness` directory contains retrospective audits that did not alter
the registered promotion gates or primary results.  The `vhd` directory is
generated from the original double-review exports by
`src/audit_vhd_annotations.py`.

Raw VideoLLM JSONL responses are intentionally not mirrored here.  Their hashes
are recorded in each confirmation archive and in the post-hoc analysis manifest.
