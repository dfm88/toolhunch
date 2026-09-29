import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "pydantic_ai_tool_search.py"


@pytest.mark.anyio
async def test_readme_example_reveals_and_executes_the_deciders_tool(fake_model: Any) -> None:
    spec = importlib.util.spec_from_file_location("pydantic_ai_tool_search_example", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    offered: list[list[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append(sorted(tool.name for tool in info.function_tools))
        if len(offered) == 1:
            return ModelResponse(parts=[ToolCallPart("search_tools", {"queries": ["weather email"]})])
        if len(offered) == 2:
            return ModelResponse(parts=[ToolCallPart("get_weather", {"city": "Milan"})])
        return ModelResponse(parts=[TextPart("Milan: sunny (demo).")])

    decision_model = fake_model({"get_weather": 10.0}, none_weight=0.1)
    agent = example.build_agent(FunctionModel(respond), decision_model=decision_model)
    result = await agent.run("What's the weather in Milan?")
    returns = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart)
    ]
    assert result.output == "Milan: sunny (demo)."
    assert [part.content["discovered_tools"] for part in returns] == [[{"name": "get_weather"}]]
    assert offered == [["search_tools"], ["get_weather", "search_tools"], ["get_weather", "search_tools"]]
    assert len(decision_model.requests) == 1
    assert "What's the weather in Milan?" in decision_model.requests[0].state
    assert set(decision_model.requests[0].questions["tool"].options) == {"get_weather", "send_email", "none"}
    assert any(
        "18 degrees Celsius (demo)" in str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )


def test_readme_python_block_is_the_executable_example() -> None:
    blocks = re.findall(r"^```python\n(.*?)^```", (ROOT / "README.md").read_text(), flags=re.MULTILINE | re.DOTALL)
    assert blocks == [EXAMPLE.read_text()]
