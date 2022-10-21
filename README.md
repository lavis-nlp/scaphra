# SCAPHRA

SpaCy component for scattered phrase matching.


## Installation

Python 3.9 is required. We recommend miniconda for python version management.

```
conda create -n scaphra python=3.9
pip install scaphra
```

## Usage

See `scaphra/example.py` for an example application.

The matcher is a single SpaCy component which matches scattered
phrases both using their lemmas and stems. This is important when the
text quality is bad and relying on lemmata does not suffice. Also, in
some languages (such as German) phrases are often non-contiguous. For
example: Matching `does not start` should match `Does it not always
start well?`.

This implementation should run reasonably fast (it uses a
state-machine which memoizes all partial matches such that each text
only needs to be traversed once). However, the computational cost
rises when many, similar patterns are applied to large texts with many
matches (runtime complexity is dependent on the number of patterns).
