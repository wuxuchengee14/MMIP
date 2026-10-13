"""Corpus BLEU-1..4 with multi-reference clipped counts, no smoothing.

Papineni et al. (2002): closest reference length (shorter on ties), corpus
brevity penalty, uniform n-gram weights. Scores are in [0,1].
"""
import collections
import math

from quiz1.Code.prepare import tokenize


def ngrams(tokens, n):
    return collections.Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(references, hypotheses):
    if len(references) != len(hypotheses) or not references:
        raise ValueError("BLEU requires aligned nonempty reference and hypothesis lists")
    clipped, totals = [0] * 4, [0] * 4
    candidate_length, reference_length = 0, 0
    for refs, hypothesis in zip(references, hypotheses):
        candidate = tokenize(hypothesis)
        ref_tokens = [tokenize(ref) for ref in refs]
        if not ref_tokens:
            raise ValueError("Missing reference captions")
        candidate_length += len(candidate)
        reference_length += min((len(r) for r in ref_tokens), key=lambda n: (abs(n-len(candidate)), n))
        for i in range(4):
            counts = ngrams(candidate, i + 1)
            maximum = collections.Counter()
            for ref in ref_tokens:
                maximum |= ngrams(ref, i + 1)
            clipped[i] += sum((counts & maximum).values())
            totals[i] += sum(counts.values())
    bp = math.exp(min(0, 1 - reference_length / candidate_length)) if candidate_length else 0.0
    precisions = [a / b if b else 0.0 for a, b in zip(clipped, totals)]
    result = {"brevity_penalty": bp, "hypothesis_tokens": candidate_length, "reference_tokens": reference_length}
    for n in range(1, 5):
        ps = precisions[:n]
        result[f"bleu{n}"] = bp * math.exp(sum(math.log(p) for p in ps)/n) if all(ps) else 0.0
    return result
