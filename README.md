# LangGraph + FastAPI Chat Demo

这是一个基于 **LangGraph** 和 **FastAPI** 实现的简易 Chat Demo，用于验证和演示 **基于图状态的对话 Agent** 在实际 Web 服务中的基本用法，包括会话管理、上下文控制以及流式响应。

该项目主要用于技术验证和结构探索，不是完整的生产级聊天系统。

---

## 特性简介

- 基于 **FastAPI** 提供 HTTP 接口
- 使用 **LangGraph** 构建对话状态图（StateGraph）
- 支持多轮对话与会话隔离
- 支持流式返回（SSE）
- 可扩展的 Agent 节点设计（如 summary、tool call、LLM call 等）
- 代码结构尽量保持简单，便于二次开发和实验