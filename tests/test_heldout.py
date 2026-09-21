import numpy as np
import pytest
from refusalclimb.heldout import evaluate

def batch(prefix):
    return dict(activations=np.array([[-1., 0], [-2., 0], [1., 0], [2., 0]]),
        labels=np.array([0, 0, 1, 1]), ids=np.array([prefix+str(i) for i in range(4)]),
        categories=np.array(["a", "b", "a", "b"]))

def test_heldout_separation():
    result = evaluate(batch("train"), batch("test"))
    assert result["overall"]["accuracy"] == 1
    assert set(result["by_category"]) == {"a", "b"}

def test_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        evaluate(batch("same"), batch("same"))

def test_threshold_not_fit_on_test_labels():
    test = batch("test")
    before = evaluate(batch("train"), test)
    test["labels"] = 1 - test["labels"]
    after = evaluate(batch("train"), test)
    assert before["threshold"] == after["threshold"]
    assert after["overall"]["accuracy"] == 0

def test_nonfinite_rejected():
    train = batch("train")
    train["activations"][0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate(train, batch("test"))
