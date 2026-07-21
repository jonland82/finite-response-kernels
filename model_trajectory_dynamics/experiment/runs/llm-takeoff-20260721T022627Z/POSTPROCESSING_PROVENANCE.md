# Post-processing provenance

The AWS training workload completed all four pilots and all six confirmation
seeds in 392.4 seconds. `results/cloud_exit_status.txt` nevertheless records
`run_exit_code=1` because the original analyzer failed after training while
writing model-specific dictionaries with different CSV columns.

The raw trajectory files and `run_summary.csv` were successfully archived and
collected before instance teardown. The analyzer was corrected locally to:

1. write the union of model-specific CSV fields;
2. add direct threshold-crossing widths;
3. use checkpoint-level BIC and cross-validated checkpoint RMSE for model
   comparison, avoiding an independence assumption across repeated test-set
   evaluations;
4. preserve the binomial likelihood/BIC only as descriptive fit information.

The final analyzer (`SHA-256
93d85d4861a7828bccae94030f926fc82724237f925f915e4f9f51d882300a5b`) was run
locally against `results/`. Its synthetic delta, finite-response, and mixture
checks passed, and `analysis_summary.json` records status `complete`.

The original collected `results.tar.gz` remains the immutable cloud archive
(`SHA-256
85cfa25be9e4eb09a29538c28d1335da7b87edcb99491b3a48acfa10da7aa03a`). Files
inside the extracted `results/` directory include the subsequent local analysis
outputs and therefore are intentionally newer than the archive.
