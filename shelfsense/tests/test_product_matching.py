import numpy as np
import pytest

from src.product_matcher import match_embedding


def test_low_similarity_unknown():
    result = match_embedding(np.array([1, 0]), {"a": [np.array([0, 1])]}, 0.85)
    assert result["product_id"] is None
    assert result["status"] == "Unknown product"


def test_ambiguity_between_distinct_products():
    result = match_embedding(np.array([1, 0]), {"a": [np.array([1, 0])], "b": [np.array([0.999, 0.01])]})
    assert result["status"] == "Needs verification"
    assert result["product_id"] is None


def test_multiple_references_do_not_create_false_ambiguity():
    result = match_embedding(np.array([1, 0]), {"a": [np.array([1, 0]), np.array([1, 0])], "b": [np.array([0, 1])]})
    assert result["product_id"] == "a"


def test_empty_catalogue():
    assert match_embedding(np.array([1, 0]), {})["status"] == "Unknown product"


def test_bad_embedding():
    with pytest.raises(ValueError):
        match_embedding(np.zeros(3), {})
