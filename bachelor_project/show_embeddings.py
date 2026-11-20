import sys

import numpy as np


def print_embeddings(
    file_path: str = "src/test-data/chunk_embeddings.npy",
    index: int = 0,
) -> None:
    """Load and print the first embedding stored in a .npy file, in full."""
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    embeddings = np.load(file_path)
    print(embeddings[index])


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print_embeddings(index=idx)
