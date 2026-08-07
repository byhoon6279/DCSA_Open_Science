# RQ2-4 Experimental Design

## One-Line Summary

RQ2-4 validates whether the instability exposed by masking and flip analyses
can be reproduced with PE-inspired feature-vector interventions that approximate
mutable static information in the EMBER representation.

## Main Design Choices

- PE-inspired, not executable-level
- malware-only perturbation by default
- important vs random comparison inside the same candidate pool
- balanced wild setting as the main analysis
- unpacked balanced setting as the robustness check

## Intended Interpretation

RQ2-4 is a validation experiment. It should not be framed as an attack
pipeline, binary rewriting system, or problem-space adversarial example
generator.
