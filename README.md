# SCAPHRA

SpaCy component for scattered phrase matching.

1. You have documents such as `In a hole in the ground there lived a Hobbit.`
1. You want to match patterns like `in holes live hobbits`
1. Then you need this spaCy component!


## Installation

Python 3.9 is required. We recommend miniconda for python version management.

```
conda create -n scaphra python=3.9
pip install scaphra
```

## Usage

```python
phrasemap = {'hobbits': ['in', 'holes', 'live', 'hobbits']}
nlp.add_pipe("scaphra", config=dict(phrasemap=phrasemap))
doc = nlp("In a hole in the ground there lived a Hobbit")
# now doc.spans contains a SpanGroup with the matched tokens
```

See `scaphra/example.py` for multiple, full examples.

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
