from typing import TypedDict,Annotated,Any,Literal

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage,ToolMessage,HumanMessage,RemoveMessage
from langgraph.graph import StateGraph, START,END
from langgraph.graph.state import CompiledStateGraph
from langgraph.config import RunnableConfig
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from langchain_core.messages.utils import count_tokens_approximately

from agent.prompt import SystemPrompt,AbstractPrompt
from agent.model import qwen3_8b,deepseek_r1_distill_qwen3_8b
from agent.tool.math_tookit import tool_map as math_tool_map
from agent.tool.time_toolkit import tool_map as time_tool_map


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    current_user_id:str
    current_user_name:str

    current_session_id:str

    current_summary: str


def create_chat_agent(mongo_client)->CompiledStateGraph:
    # 模型加工具
    qwen3_8b_with_tools = qwen3_8b.bind_tools([*list(time_tool_map.values()), *list(math_tool_map.values())])
    # 工具地图
    tool_map = {**time_tool_map, **math_tool_map}

    async def runtime_load_node(state:AgentState,config:RunnableConfig)->dict[str,Any]:
        runtime_config = config["configurable"]
        # f"{user_id}-{session_id}"
        user_info = runtime_config["thread_id"].split("-")
        return {
            "current_user_id":user_info[0],
            "current_user_name":f"用户{user_info[0]}",
            "current_session_id":user_info[1],
            "current_summary":"",
        }

    async def delete_tool_message_node(state: AgentState)->dict[str,Any]:
        delete_messages = [
            RemoveMessage(id=m.id) for m in state["messages"] if (
                    m.type == "tool" or (hasattr(m,"tool_calls") and getattr(m,"tool_calls"))
            )
        ]
        return {"messages": delete_messages}

    async def summarize_conversation_node(state: AgentState)->dict[str,Any]:
        tokens = count_tokens_approximately(state["messages"])

        if tokens < 128 * 1000:
            return dict()

        # First, we get any existing summary
        summary = state.get("current_summary", "")

        # Create our summarization prompt
        if summary:

            # A summary already exists
            summary_message = (
                f"This is a summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )

        else:
            summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = await qwen3_8b.ainvoke(messages)

        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"current_summary": response.content, "messages": delete_messages}

    async def llm_call_node(state:AgentState)->dict[str,Any]:
        basic_messages = [SystemPrompt]
        current_summary = state.get("current_summary", "")
        if current_summary:
            basic_messages.append(SystemMessage(
                content=f"因对话超过模型上下文，因此做了一次摘要总结，"
                        f"以下是对话历史的摘要，请作为已知事实使用，而不是逐字复述: {current_summary}")
            )

        return {
            "messages":  [await qwen3_8b_with_tools.ainvoke(basic_messages + state["messages"])],
        }


    async def tool_node(state:AgentState)->dict[str,Any]:
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tool_map[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}

    async def should_continue(state: AgentState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END

    # Build workflow
    agent_builder = StateGraph(AgentState)

    # Add nodes
    agent_builder.add_node("llm_call_node", llm_call_node)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("runtime_load_node", runtime_load_node)
    agent_builder.add_node("summarize_conversation_node",summarize_conversation_node)
    agent_builder.add_node("delete_tool_message_node",delete_tool_message_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "runtime_load_node")
    agent_builder.add_edge("runtime_load_node", "delete_tool_message_node")
    agent_builder.add_edge("delete_tool_message_node", "summarize_conversation_node")
    agent_builder.add_edge("summarize_conversation_node","llm_call_node")
    agent_builder.add_conditional_edges(
        "llm_call_node",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call_node")

    # Compile the agent
    return agent_builder.compile(
        checkpointer=AsyncMongoDBSaver(client=mongo_client),
    )


class AbstractState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def create_abstract_agent()->CompiledStateGraph:

    return create_agent(
        model=deepseek_r1_distill_qwen3_8b,
        system_prompt=AbstractPrompt,
    )
