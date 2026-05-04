import datetime as dt
import uuid

from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage


def has_tool_result(messages):
    return any(isinstance(message, ToolMessage) for message in messages)


def bind_tools_require_first_call(llm, tools, messages):
    """Require analyst tools until the first tool result is present."""
    if has_tool_result(messages):
        return llm.bind_tools(tools)
    return llm.bind_tools(tools, tool_choice="required")


def force_tool_call_if_missing(result, messages, tool_name, args):
    if result.tool_calls or has_tool_result(messages):
        return result
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": args,
                "id": f"forced_{tool_name}_{uuid.uuid4().hex[:8]}",
            }
        ],
    )


def start_date_for(current_date, days):
    end = dt.date.fromisoformat(current_date)
    return (end - dt.timedelta(days=days)).isoformat()
