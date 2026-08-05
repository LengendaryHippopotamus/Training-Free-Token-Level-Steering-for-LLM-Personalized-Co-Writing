# Dataset provenance

## CDN

The files under `data/CDN/` contain a cleaned and processed subset derived from [IBM Project CodeNet](https://github.com/IBM/Project_CodeNet):

- `reference.json`: 75 reference texts used to construct personalized guidance.
- `question.json`: 75 corresponding evaluation texts.

Project CodeNet aggregates programming-problem and submission data originating from AIZU Online Judge and AtCoder. IBM distributes the Project CodeNet repository under the Apache License 2.0 and documents the original data sources in its repository. This repository does not claim ownership of the underlying upstream material.

The released files have been further selected, cleaned, and transformed for the experiments accompanying *Training-Free Token-Level Steering for LLM Personalized Co-Writing*.

Users are responsible for reviewing and complying with the applicable Project CodeNet and original-source terms when using or redistributing these data.

## References

- IBM Project CodeNet: <https://github.com/IBM/Project_CodeNet>
- Project CodeNet paper: <https://arxiv.org/abs/2105.12655>
- AIZU Online Judge: <https://onlinejudge.u-aizu.ac.jp/>
- AtCoder: <https://atcoder.jp/>
