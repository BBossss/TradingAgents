import inspect

from cli import utils
from tradingagents.default_config import DEFAULT_CONFIG


def test_default_config_remains_openai_api_key_path():
    assert DEFAULT_CONFIG["llm_provider"] == "openai"
    assert DEFAULT_CONFIG["backend_url"] is None


def test_cli_provider_selection_keeps_openai_first_and_codex_opt_in():
    source = inspect.getsource(utils.select_llm_provider)
    provider_block = source.split("PROVIDERS = [", 1)[1].split("]", 1)[0]
    provider_lines = [
        line.strip()
        for line in provider_block.splitlines()
        if line.strip().startswith("(")
    ]

    assert provider_lines[0].startswith('("OpenAI", "openai"')
    assert any('"codex_oauth"' in line for line in provider_lines[1:])
