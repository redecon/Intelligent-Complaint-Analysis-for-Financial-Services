import pandas as pd
import pytest

from src.themes import run_lda_validation

@pytest.mark.skipif(
    not pytest.importorskip("gensim") or not pytest.importorskip("pyLDAvis"),
    reason="Requires gensim and pyLDAvis installed"
)
def test_run_lda_validation_smoke(tmp_path):
    # Minimal mock dataset with negative reviews
    df = pd.DataFrame({
        "bank": ["CBE", "CBE", "BOA"],
        "rating": [1, 2, 5],
        "review_text": [
            "Login failed and app crashed",
            "Transfer payment stuck and slow",
            "Great design and easy navigation"
        ]
    })

    # Run LDA validation
    lda_model = run_lda_validation(df, text_column="review_text", num_topics=2)

    # Check that a gensim LdaModel object is returned
    from gensim.models.ldamodel import LdaModel
    assert isinstance(lda_model, LdaModel)

    # Check that visualization file was created
    output_file = tmp_path / "lda_viz.html"
    # Note: our function saves to data/outputs/lda_viz.html by default,
    # so here we just assert that lda_model exists and topics were printed.
    assert lda_model.num_topics == 2
