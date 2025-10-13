"""Tests for Version 2 rule-based classifier."""

from scripts.classify import Version2RuleEngine


def classify(text: str):
    return Version2RuleEngine().classify(text)


def test_random_package_not_a_problem():
    result = classify("Randomly got a package in the mail yesterday.")
    assert result.is_problem == "0"
    assert result.is_software_solvable == "0"
    assert result.is_external == "0"


def test_entry_level_job_rant_problem_not_software():
    text = "Entry-level job market rant and my advice for others looking for help."
    result = classify(text)
    assert result.is_problem == "1"
    assert result.is_software_solvable == "0"
    assert result.is_external == "0"


def test_architecture_software_request_problem_not_software():
    text = "Looking for beginner architecture software that I can afford in a small apartment."
    result = classify(text)
    assert result.is_problem == "1"
    assert result.is_software_solvable == "0"
    assert result.is_external == "0"


def test_samsung_update_external():
    text = "After the Samsung update my elderly mom can't use her phone anymore."
    result = classify(text)
    assert result.is_problem == "1"
    assert result.is_software_solvable == "1"
    assert result.is_external == "1"


def test_printer_purchase_external():
    text = "Which 3D printer should I buy for learning at home?"
    result = classify(text)
    assert result.is_problem == "1"
    assert result.is_software_solvable == "0"
    assert result.is_external == "1"
