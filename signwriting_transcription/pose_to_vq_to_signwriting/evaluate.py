import argparse
from functools import lru_cache

from signwriting.tokenizer import SignWritingTokenizer
from signwriting_evaluation.metrics.bleu import SignWritingBLEU
from signwriting_evaluation.metrics.chrf import SignWritingCHRF
from signwriting_evaluation.metrics.clip import SignWritingCLIPScore
from signwriting_evaluation.metrics.similarity import SignWritingSimilarityMetric


@lru_cache(maxsize=1)
def get_metrics():
    return [
        SignWritingBLEU(),
        SignWritingCHRF(),
        SignWritingSimilarityMetric(),
        SignWritingCLIPScore(),
    ]


def load_file(file_path: str):
    tokenizer = SignWritingTokenizer()
    with open(file_path, 'r', encoding="utf-8") as file:
        signs = list(file.read().splitlines())
        return [tokenizer.tokens_to_text(sign.split(" ")) for sign in signs]


def evaluate(hypotheses: str, reference: str):
    hypotheses = load_file(hypotheses)
    references = load_file(reference)

    assert len(hypotheses) == len(references), "Hypothesis and reference must have the same number of instances"

    for metric in get_metrics():
        score = metric.corpus_score(hypotheses=hypotheses, references=[references]) * 100
        print(metric.name, f"{score:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.hypothesis, args.reference)


if __name__ == "__main__":
    main()
