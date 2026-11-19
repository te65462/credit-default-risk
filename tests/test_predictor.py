"""Input checks, data cleaning and the flag decision in predictor.py."""
import json

import pytest

import predictor


def test_valid_client_gets_a_risk_and_a_decision(example):
    result = predictor.predict(example)
    assert 0 < result["risk"] < 1
    assert result["flagged"] == (result["risk"] >= predictor.THRESHOLD)
    assert result["threshold"] == pytest.approx(0.2664, abs=5e-5)
    assert result["notes"] == []


def test_numbers_given_as_text_are_accepted(example):
    as_text = {name: str(value) for name, value in example.items()}
    assert predictor.predict(as_text)["risk"] == predictor.predict(example)["risk"]


@pytest.mark.parametrize("name, code, same_as", [("EDUCATION", 0, 4), ("EDUCATION", 5, 4),
                                                 ("EDUCATION", 6, 4), ("MARRIAGE", 0, 3)])
def test_undocumented_codes_are_cleaned_as_in_training(example, name, code, same_as):
    cleaned = predictor.predict({**example, name: same_as})
    raw = predictor.predict({**example, name: code})
    assert raw["risk"] == cleaned["risk"]
    assert any("undocumented code" in note for note in raw["notes"])


def test_flag_follows_the_threshold():
    examples = predictor.INFO["examples"]
    low, high = examples[0], examples[2]
    assert low["module8_risk"] < predictor.THRESHOLD < high["module8_risk"]
    assert predictor.predict(low["values"])["flagged"] is False
    assert predictor.predict(high["values"])["flagged"] is True


def test_missing_inputs_are_all_listed(example):
    del example["AGE"], example["PAY_AMT6"]
    with pytest.raises(predictor.InputError) as err:
        predictor.predict(example)
    assert err.value.problems == ["Age: missing", "Amount paid, April 2005: missing"]


@pytest.mark.parametrize("name, value, message", [
    ("LIMIT_BAL", -5000, "must be at least 1"),
    ("LIMIT_BAL", "ten thousand", "is not a number"),
    ("AGE", 17, "must be at least 18"),
    ("AGE", 35.5, "must be a whole number"),
    ("SEX", 3, "is not a valid code"),
    ("EDUCATION", 7, "is not a valid code"),
    ("PAY_1", 9, "is not a valid code"),
    ("PAY_1", -3, "is not a valid code"),
    ("PAY_AMT1", -1, "must be at least 0"),
    ("BILL_AMT1", float("inf"), "must be a finite number"),
    ("BILL_AMT1", True, "is not a number"),
])
def test_invalid_inputs_are_rejected(example, name, value, message):
    with pytest.raises(predictor.InputError) as err:
        predictor.predict({**example, name: value})
    assert len(err.value.problems) == 1 and message in err.value.problems[0]


def test_unknown_input_names_are_rejected(example):
    example["PAY_0"] = example.pop("PAY_1")  # the raw file's name for PAY_1
    with pytest.raises(predictor.InputError) as err:
        predictor.predict(example)
    assert err.value.problems == ["unknown input name(s): PAY_0", "Repayment status, September 2005: missing"]


def test_values_outside_training_range_are_scored_with_a_note(example):
    result = predictor.predict({**example, "AGE": 85})
    assert 0 < result["risk"] < 1
    assert result["notes"] == ["Age: 85 is outside the range seen in training (21 to 79), "
                               "so this estimate is less certain."]


def test_status_code_unseen_for_that_month_gets_a_note(example):
    result = predictor.predict({**example, "PAY_5": 1})  # code 1 never occurs in May or April
    assert any("never occurs for this month" in note for note in result["notes"])


def test_json_bridge_used_by_the_web_page(example):
    ok = json.loads(predictor.predict_json(json.dumps(example)))
    assert ok["ok"] is True and ok["risk"] == predictor.predict(example)["risk"]
    bad = json.loads(predictor.predict_json(json.dumps({**example, "SEX": 9})))
    assert bad == {"ok": False, "problems": ["Sex: 9 is not a valid code (1, 2)"]}
