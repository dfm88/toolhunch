from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolSearchReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset

from toolhunch.integrations.pydantic_ai import catalog_from_tool_defs
from toolhunch_bench.baselines import KeywordsRetriever

pytestmark = pytest.mark.anyio


def tool(name: str, description: str | None = None) -> ToolDefinition:
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    return ToolDefinition(name=name, description=description, parameters_json_schema=schema, defer_loading=True)


TOOLS = [
    tool("get_weather", "Get the weather forecast for a city."),
    tool("get_forecast", "Weather forecast, hourly."),
    tool("WeatherAlerts", "Severe WEATHER alerts."),
    tool("send_email", "Send an email message."),
    tool("send_sms", "Send an SMS message to a phone number."),
    tool("sendSlackMessage", "Post a message to Slack."),
    tool("read_email", "Read email messages from the inbox."),
    tool("list_calendar_events", "List the events in a calendar."),
    tool("create_calendar_event", "Create an event in a calendar."),
    tool("delete_calendar_event", ""),
    tool("get_diary_day"),
    tool("search_web", "Search the web."),
    tool("search_news", "Search news articles."),
    tool("search_images", "Search the web for images."),
    tool("translate_text", "Translate text between languages."),
    tool("detect_language", "Detect the language of a text."),
    tool("convert_currency", "Convert an amount between currencies."),
    tool("get_exchange_rate", "Exchange rate between two currencies."),
    tool("stock_price", "Get the stock price for a ticker."),
    tool("crypto_price", "Get the price of a crypto currency."),
    tool("ocr_image", "Read the text in an image."),
    tool("resize_image", "Resize an image."),
    tool("image2text", "Describe an image in text."),
    tool("text2speech", "Turn text into speech."),
    tool("speech2text", "Transcribe speech to text."),
    tool("get_user", "Get a user by id."),
    tool("get_user_v2", "Get a user by ID, version 2."),
    tool("list_users", "List users."),
    tool("http_get", "HTTP GET request."),
    tool("http_post", "HTTP POST request."),
]

QUERIES = [
    ["weather forecast"],
    ["send a message", "email"],
    ["calendar event"],
    ["User ID", "get user v2"],
    ["image to text", "text2speech"],
    ["???"],
]


async def names_found_by_pydantic_ai(queries: list[str]) -> list[str]:
    """The public path: a model calls `search_tools` under `ToolSearch(strategy="keywords")`."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("search_tools", {"queries": queries})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(model),
        toolsets=[ExternalToolset(TOOLS)],
        capabilities=[ToolSearch(strategy="keywords", max_results=1000)],
    )
    result = await agent.run("find tools")
    [found] = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart)
    ]
    return [match["name"] for match in found.content["discovered_tools"]]


@pytest.mark.parametrize("queries", QUERIES)
async def test_keywords_retriever_matches_pydantic_ai(queries: list[str]) -> None:
    catalog = catalog_from_tool_defs(TOOLS)
    retriever = KeywordsRetriever(order=[t.name for t in TOOLS])  # upstream searches in registration order

    result = await retriever.retrieve(queries, catalog, k=len(catalog))

    assert [match.card.id for match in result.matches] == await names_found_by_pydantic_ai(queries)
