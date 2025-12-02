import os
import pytest
from omero_screen.config import getenv_as_int, getenv_as_bool

def test_getenv_as_int_basic(monkeypatch):
    monkeypatch.setenv("TEST_INT", "123")
    assert getenv_as_int("TEST_INT", 0) == 123

def test_getenv_as_int_with_comment(monkeypatch):
    monkeypatch.setenv("TEST_INT", "123 # comment")
    assert getenv_as_int("TEST_INT", 0) == 123

def test_getenv_as_int_with_comment_no_space(monkeypatch):
    monkeypatch.setenv("TEST_INT", "123#comment")
    assert getenv_as_int("TEST_INT", 0) == 123

def test_getenv_as_int_invalid(monkeypatch):
    monkeypatch.setenv("TEST_INT", "invalid")
    assert getenv_as_int("TEST_INT", 10) == 10

def test_getenv_as_int_missing(monkeypatch):
    monkeypatch.delenv("TEST_INT", raising=False)
    assert getenv_as_int("TEST_INT", 10) == 10

def test_getenv_as_bool_with_comment(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "true # comment")
    assert getenv_as_bool("TEST_BOOL") is True

def test_getenv_as_bool_with_comment_no_space(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "true#comment")
    assert getenv_as_bool("TEST_BOOL") is True
