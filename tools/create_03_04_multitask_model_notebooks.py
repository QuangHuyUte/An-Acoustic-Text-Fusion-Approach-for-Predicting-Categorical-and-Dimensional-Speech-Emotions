"""Compatibility wrapper.

The current project direction uses the full Emotion2Vec + acoustic two-branch
pipeline. This wrapper prevents the old metadata-only generator from being
used accidentally.
"""

from create_full_iemocap_emotion2vec_pipeline import main


if __name__ == "__main__":
    main()
