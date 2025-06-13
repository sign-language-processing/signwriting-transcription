# Inference Strategies

## Overview

The decoding strategy depends heavily on our compute budget. 
While SignWriting theoretically allows infinite possible signs, most lexical signs follow an 80:20 distribution pattern 
with similar signs repeating frequently.

## Decoding Approaches

### 0. Direct Model Decoding

- Use standard beam search or greedy decoding directly from the model
- No additional constraints or vocabulary restrictions
- **Enhancement**: Integrate probabilities from a sign language translation model (text-to-signwriting) to guide decoding
  (Only applicable when the label is available and we are not using this to test accuracy)
- Fast and simple, but may produce invalid or suboptimal signs

### 1. Exhaustive Vocabulary Decoding

- Maintain a large vocabulary of all known signs
- Force decode all signs in vocabulary
- Select the highest probability candidate
- **Limitation**: Vocabulary grows continuously, increasing compute requirements

### 2. Beam Search + Nearest Neighbor Refinement

This hybrid approach balances quality and efficiency:

1. **Initial Decode**: Use beam search to get model predictions
2. **Neighbor Search**: Find nearest neighbors in existing sign corpus
3. **Force Decode**: Decode top N closest signs (e.g., top 100)
4. **Iterative Refinement**: Repeat until no better candidates found

**Benefits**: 
- Can only improve predictions (assuming reliable model probabilities)
- Computationally tractable
- Leverages existing sign corpus

## Nearest Neighbor Implementation

### Sign Hashing Strategy
- Convert signs to "A" string representation
- Report symbols sorted by symbol ID
- Enables efficient string-based search

### Search Process
1. Hash target sign to "A" string format
2. Perform string search to find closest buckets
3. Apply SignWriting similarity metrics to candidates
4. Return top matches for force decoding

This approach provides a practical balance between decode quality and computational efficiency.