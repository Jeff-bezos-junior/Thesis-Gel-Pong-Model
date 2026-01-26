# Retention Model

This directory contains the retention-based model corresponding to Chapter 4.2
of the thesis.

In this model, the Pong control framework is identical to the main model:
three region-wise current values are mapped to paddle position using the same
quadratic fitting and control logic. However, unlike the main model, the time
evolution of the current is not designed to induce learning-like performance
improvement.

Instead, this model assigns current dynamics that emphasize **retention and
stability**. Current responses are carried over between successive stimuli,
resulting in smooth and consistent temporal behavior. As a consequence,
region-wise current differences do not systematically expand over time, and
performance improvement in Pong is not expected to accumulate.

This model is used to examine how robust and physically plausible current
properties—such as slow decay and continuity—affect Pong behavior, and to
contrast these effects with learning-inducing conditions analyzed in the
main model.

## Related Thesis Section
- Chapter 4.2: 電流保持特性に着目した補助的モデル

## Key Characteristics
- Same Pong control logic as the main model
- Region-wise currents evolve with retention across stimuli
- No mechanism to enforce learning or performance improvement
- Used for interpretation and comparison, not optimization

## How to Run
```bash
python retention_model.py
