import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _env(name: str) -> str | None:
    """A variable from the environment or the repository's .env."""
    load_dotenv(Path(__file__).parents[1] / ".env", override=False)
    return os.environ.get(name)


@pytest.fixture
def openai_api_key() -> str:
    """The OpenAI key from the environment or the repository's .env; skips the test when absent."""
    if not (key := _env("OPENAI_API_KEY")):
        pytest.skip("OPENAI_API_KEY is not set")
    return key


@pytest.fixture
def typesafe_api_key() -> str:
    """The TypeSafe key from the environment or the repository's .env; skips the test when absent."""
    if not (key := _env("TYPESAFE_API_KEY")):
        pytest.skip("TYPESAFE_API_KEY is not set")
    return key


@pytest.fixture
def clm_base_url() -> str:
    """The CLM server root from the environment or the repository's .env; skips the test unless the key is set too."""
    base_url, key = _env("CLM_BASE_URL"), _env("CLM_API_KEY")
    if not base_url or not key:
        pytest.skip("CLM_BASE_URL and CLM_API_KEY are not both set")
    return base_url
