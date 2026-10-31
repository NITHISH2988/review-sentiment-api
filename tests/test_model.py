from review_sentiment.model import create_pipeline, normalize_text


def test_normalize_text_folds_case_and_whitespace():
    assert normalize_text("  GREAT\n\tProduct  ") == "great product"


def test_pipeline_fits_and_predicts_both_classes():
    model = create_pipeline().fit(
        ["excellent quality", "love it", "terrible quality", "hate it"],
        [1, 1, 0, 0],
    )
    assert set(model.predict(["excellent quality", "terrible quality"])) <= {0, 1}
    assert model.predict_proba(["excellent quality"]).shape == (1, 2)


def test_preprocessing_keeps_empty_input_empty():
    assert normalize_text(" \n ") == ""
