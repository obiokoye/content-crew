import pytest

from content_crew.config import load_settings, make_llm
from content_crew.demo import ScriptedLLM


def test_demo_mode_when_no_key():
    settings = load_settings({"LLM_PROVIDER": "gemini"})
    assert settings.demo and "GEMINI_API_KEY" in settings.reason
    assert isinstance(make_llm(settings), ScriptedLLM)


def test_demo_mode_flag_wins_over_a_key():
    assert load_settings({"GEMINI_API_KEY": "x", "DEMO_MODE": "true"}).demo


def test_real_provider_settings():
    settings = load_settings({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "k", "LLM_MODEL": "m"})
    assert not settings.demo
    assert settings.provider.base_url == "https://api.groq.com/openai/v1"
    assert settings.label == "groq · m"
    llm = make_llm(settings)
    assert llm.model == "m"  # built, but never called in tests


def test_unknown_provider_exits():
    with pytest.raises(SystemExit):
        load_settings({"LLM_PROVIDER": "nope"})
