# Results for the current paper

The two confirmation directories contain the fixed-protocol held-out result
archives used in the paper. Each includes an `archive.sha256` manifest. The
historical `TRACE-RC` identifier remains in filenames, columns, and schema
values so that archived checksums continue to verify.

The `posthoc_robustness` directory contains secondary analyses that leave the
main target-pilot decisions and primary results unchanged. It includes
model-level metrics and paired complete-video bootstrap comparisons for seven
training-free confidence and consistency scores, together with fixed-coverage
risk, pilot-size, and compute-budget analyses. The `vhd` directory is generated
from the original double-review exports by `src/audit_vhd_annotations.py`.

Raw VideoLLM JSONL responses are not mirrored here. Their hashes are preserved
in the manifests, and the included scored predictions are sufficient to
reconstruct the reported held-out metrics and selection decisions.
