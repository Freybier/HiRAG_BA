"""RAGAS evaluation tooling for the bachelor-project RAG systems.

Two decoupled workflows:

* collectors (one per system, each in that system's venv):
  ``collect_hirag.py``, ``collect_naive.py``, ``collect_test_data.py`` run the
  questions and save answers plus the exact context the answer LLM received.
* ``score.py`` (evaluation venv with the local RAGAS checkout): score a saved
  run with RAGAS metrics without re-running the RAG system.

See ``README.md`` in this folder for setup and usage.
"""
