import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def openai_api_key() -> str:
    """The OpenAI key from the environment or the repository's .env; skips the test when absent."""
    load_dotenv(Path(__file__).parents[1] / ".env", override=False)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY is not set")
    return key
