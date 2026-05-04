from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.social_media_analyst import create_social_media_analyst


class RecordingLLM:
    def __init__(self):
        self.bound_tools = None
        self.bound_kwargs = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        self.bound_kwargs = kwargs
        return RunnableLambda(lambda _messages: AIMessage(content="placeholder"))


def _state():
    return {
        "messages": [HumanMessage(content="GLD")],
        "company_of_interest": "GLD",
        "trade_date": "2026-05-04",
    }


def _state_after_tool_result():
    return {
        "messages": [
            HumanMessage(content="GLD"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_stock_data",
                        "args": {"ticker": "GLD", "curr_date": "2026-05-04"},
                        "id": "call_1",
                    }
                ],
            ),
            ToolMessage(content="price csv", tool_call_id="call_1"),
        ],
        "company_of_interest": "GLD",
        "trade_date": "2026-05-04",
    }


def test_market_analyst_requires_a_tool_call_on_first_pass():
    llm = RecordingLLM()

    result = create_market_analyst(llm)(_state())

    assert llm.bound_tools
    assert llm.bound_kwargs["tool_choice"] == "required"
    assert result["messages"][0].tool_calls[0]["name"] == "get_stock_data"


def test_social_media_analyst_requires_a_tool_call_on_first_pass():
    llm = RecordingLLM()

    result = create_social_media_analyst(llm)(_state())

    assert llm.bound_tools
    assert llm.bound_kwargs["tool_choice"] == "required"
    assert result["messages"][0].tool_calls[0]["name"] == "get_news"


def test_news_analyst_requires_a_tool_call_on_first_pass():
    llm = RecordingLLM()

    result = create_news_analyst(llm)(_state())

    assert llm.bound_tools
    assert llm.bound_kwargs["tool_choice"] == "required"
    assert result["messages"][0].tool_calls[0]["name"] == "get_news"


def test_fundamentals_analyst_requires_a_tool_call_on_first_pass():
    llm = RecordingLLM()

    result = create_fundamentals_analyst(llm)(_state())

    assert llm.bound_tools
    assert llm.bound_kwargs["tool_choice"] == "required"
    assert result["messages"][0].tool_calls[0]["name"] == "get_fundamentals"


def test_market_analyst_allows_final_report_after_tool_result():
    llm = RecordingLLM()

    create_market_analyst(llm)(_state_after_tool_result())

    assert llm.bound_tools
    assert "tool_choice" not in llm.bound_kwargs
