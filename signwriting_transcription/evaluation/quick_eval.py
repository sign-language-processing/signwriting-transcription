import pandas as pd
from signwriting_evaluation.metrics.similarity import SignWritingSimilarityMetric

TSV_PATH = "/scratch/amoryo/tmp/signwriting-transcription/data.test.tsv"
GENERATED_PATH = "/home/amoryo/sign-language/signwriting-transcription/test.output.txt"

def main():
    # Load TSV and get output column
    df = pd.read_csv(TSV_PATH, sep='\t')
    expected_outputs = df['output'].tolist()

    # Load generated text
    with open(GENERATED_PATH) as f:
        generated_outputs = [line.strip() for line in f.readlines()]

    # Initialize similarity metric
    metric = SignWritingSimilarityMetric()

    corpus_score = metric.corpus_score(generated_outputs, [expected_outputs])
    print(f"Corpus similarity score: {corpus_score:.4f}")

    # Calculate scores line by line
    scores = []
    for i, (expected, generated) in enumerate(zip(expected_outputs, generated_outputs)):
        score = metric.score(expected, generated)
        scores.append((i, score, expected, generated))

    # Sort by score (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)
    print("Average similarity score:", sum(score for _, score, _, _ in scores) / len(scores))

    # Remove all scores where 'encoder_prompt': '__pose__'
    scores = [(line_idx, score, expected, generated) for line_idx, score, expected, generated in scores
              if df.iloc[line_idx]['encoder_prompt'] != '__pose__']
    print("Average similarity score (no Chicago):", sum(score for _, score, _, _ in scores) / len(scores))

    # Remove all scores equal to 1
    scores = [(line_idx, score, expected, generated) for line_idx, score, expected, generated in scores
              if score < .7]
    print("Average similarity score (no perfect scores):", sum(score for _, score, _, _ in scores) / len(scores))

    # scores where the line "start" is not 0
    segmented_scores = [
        (line_idx, score, expected, generated) for line_idx, score, expected, generated in scores
        if df.iloc[line_idx]['signal_start'] != 0
    ]
    print("Average similarity score (segmented):",
          sum(score for _, score, _, _ in segmented_scores) / len(segmented_scores))


    # Print top 10
    print("Top 10 highest similarity scores:")
    for i, (line_idx, score, expected, generated) in enumerate(scores[:10]):
        print(f"{i+1}. Line {line_idx}: Score {score:.4f}")
        print(f"   Expected: {expected}")
        print(f"   Generated: {generated}")
        # print(f"   Row: {df.iloc[line_idx].to_dict()}")
        print()

if __name__ == "__main__":
    main()

