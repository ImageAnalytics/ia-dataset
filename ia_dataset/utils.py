import numpy as np

def find_continuous_sequences(arr, val):
    sequences = []
    current_sequence = []

    for idx, value in enumerate(arr):
        if value == val:
            current_sequence.append(idx)
        elif current_sequence:
            sequences.append(current_sequence)
            current_sequence = []

    if len(current_sequence) > 0:
        sequences.append(current_sequence)

    return sequences