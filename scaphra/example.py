# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import spacy

import scaphra.matcher  # noqa

nlp = spacy.load(
    "de_core_news_sm",
    disable=[
        "tok2vec",
        "tagger",
        "ner",
        "parser",
        "attribute_ruler",
    ],
)

# register the phrases to be matched as a simple dictionary
# multiple constraints apply to the phrasemap due to spacy
# (must be json serializable):
#  (1) no integer keys
#  (2) no sets/tuples

phrasemap = {
    "single": ["drucken"],
    "double": ["druckt nicht", "wird getroffen"],
    "triple": ["will nicht starten"],
}


def match_and_print(nlp, string):
    doc = nlp(string)
    print("\n", string)
    for key, group in doc.spans.items():
        print(f"  matched {key=} ({group})")

        for span in group:
            print(f"    label={span.label_} ({span.start}-{span.end})")


def example1():
    print("\nEXAMPLE 1: scattered phrase matching")

    nlp.add_pipe(
        "scaphra",
        config=dict(
            phrasemap=phrasemap,
        ),
    )

    # now the pipeline can be used to process documents.
    # the resulting doc objects now have a .spans property
    # which holds all matches as a dictionary:
    #   - the KEYS of the dict are strings of format "match:KEY:POSITIONS"
    #     where KEY is the key given in the phrasemap
    #     and the POSITIONS are the .-separated token start indexes
    #     (this is just to create unique identifier)
    #   - the VALUES are spacy.tokens.SpanGroup instances which group
    #     a single match with multiple single-token spans

    match_and_print(nlp, "das will nicht drucken!")
    match_and_print(nlp, "das wird getroffen")
    match_and_print(nlp, "montags druckt die kiste nicht")
    match_and_print(nlp, "der motor will einfach nicht richtig starten")


example1()


# max_space
def example2():
    print("\nEXAMPLE 2: max_space constraint")

    nlp.remove_pipe("scaphra")
    nlp.add_pipe(
        "scaphra",
        config=dict(
            phrasemap=phrasemap,
            max_space=2,
        ),
    )

    match_and_print(nlp, "das wird getroffen")
    match_and_print(nlp, "das wird nicht getroffen")


example2()
