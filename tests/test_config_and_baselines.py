"""Tests for the .env loader and the lexicon baseline (the FinBERT baseline needs a model download and is not tested here)."""
import os

import pytest

from src import config
from src.baselines import lexicon_score


def test_parse_env_lines():
    text = "# comment\nA=1\nexport B=\"two words\"\nC=\nD = x \nno equals sign\n"
    assert config._parse(text) == {"A": "1", "B": "two words", "D": "x"}      # empty values mean 'not filled in yet'


def test_load_env_never_overrides_the_shell_and_handles_a_bom(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_bytes("﻿FS_TEST_KEY=from_file\nFS_TEST_NEW=new\n".encode("utf-8"))      # Notepad adds a BOM
    monkeypatch.setenv("FS_TEST_KEY", "from_shell")
    monkeypatch.delenv("FS_TEST_NEW", raising=False)
    try:
        loaded = config.load_env(env_file)
        assert os.environ["FS_TEST_KEY"] == "from_shell"
        assert os.environ["FS_TEST_NEW"] == "new" and loaded == ["FS_TEST_NEW"]
    finally:
        os.environ.pop("FS_TEST_NEW", None)


def test_missing_env_file_is_fine(tmp_path):
    assert config.load_env(tmp_path / "nope.env") == []


def test_get_secret_error_never_contains_a_value(monkeypatch):
    monkeypatch.delenv("FS_NOT_SET", raising=False)
    with pytest.raises(RuntimeError) as exc:
        config.get_secret("FS_NOT_SET")
    assert "FS_NOT_SET" in str(exc.value)
    monkeypatch.setenv("FS_SET", "supersecretvalue")
    assert "supersecretvalue" not in str(config.secret_status("FS_SET"))


def test_lexicon_score_signs_and_edge_cases():
    pos, neg = {"GAINS", "STRONG"}, {"LOSSES", "WEAK"}
    scores = lexicon_score(["Strong gains", "Weak sales and losses", "Annual meeting", "", "gains and LOSSES"], pos, neg)
    assert scores[0] == 1.0                          # 2 positive words out of 2
    assert scores[1] == pytest.approx(-2 / 4)        # 2 negative words out of 4 tokens
    assert scores[2] == 0.0 and scores[3] == 0.0     # no dictionary words / empty text
    assert scores[4] == 0.0                          # one of each cancels, and matching ignores case
