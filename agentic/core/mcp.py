"""MCP (Model Context Protocol) integration abstractions.

This module adds a thin, deployment无关的适配层，让 MCP Server 暴露的工具
可以作为 `Tool` 在 agentic 框架中被复用。

设计目标：
- **模型无关**：只关心调用 MCP tool，不耦合具体 LLM
- **部署无关**：不假设传输层，既可以用官方 `mcp` SDK，也可以自实现客户端
- **工具解耦**：通过 `MCPTool` 适配到统一的 `Tool` 接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, Field

from agentic.core.tool import Tool, ToolSchema


class MCPClient(ABC):
    """抽象的 MCP 客户端接口。

    你可以：
    - 基于官方 `mcp` Python SDK 实现一个子类
    - 或者用任意 RPC / HTTP 协议实现自己的客户端
    """

    @abstractmethod
    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        auth: Optional["MCPAuthConfig"] = None,
    ) -> Any:
        """Call a tool exposed by an MCP server.

        Args:
            server_name: MCP server 标识（例如配置名）
            tool_name: MCP tool 名称
            arguments: 传给 MCP tool 的参数
        """
        raise NotImplementedError

    def stream_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        auth: Optional["MCPAuthConfig"] = None,
    ) -> Iterable[Any]:
        """Stream results from an MCP tool (可选实现).

        实现方可以：
        - 基于 MCP 协议的 streamable 能力，逐步 yield chunk
        - 或者在不支持流式时，直接 `yield self.call_tool(...)`
        """
        raise NotImplementedError

    def get_tool_schema(
        self,
        server_name: str,
        tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        """可选：从 MCP server 拉取指定 tool 的 JSON Schema。

        默认实现返回 None；如果你有能力从 MCP 协议中获取 schema，
        可以在自定义客户端里 override 这个方法。
        """
        return None


class MCPAuthConfig(BaseModel):
    """MCP 认证配置（抽象，不绑定具体协议）。"""

    # 如 "bearer" / "basic" / "api_key" / "custom"
    auth_type: str = Field(..., description="Authentication type identifier")
    # 常见场景：Bearer Token / API Key 等
    token: Optional[str] = Field(
        None,
        description="Optional access token / API key when applicable",
    )
    # 也允许直接指定 header / 其他元信息，由 MCPClient 自行解释
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP-like headers for auth, if applicable",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional auth metadata interpreted by MCPClient",
    )


class MCPToolConfig(BaseModel):
    """配置一个 MCPTool。"""

    server_name: str = Field(..., description="MCP server identifier (e.g., configured name)")
    tool_name: str = Field(..., description="Tool name exposed by the MCP server")
    description: str = Field(..., description="High-level description of the MCP tool")
    # 当 MCPClient 无法提供 schema 时，可以在这里手动指定 JSON Schema
    parameters_schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        },
        description="Optional JSON schema for tool parameters",
    )
    auth: Optional[MCPAuthConfig] = Field(
        default=None,
        description="Optional auth config passed through to MCPClient",
    )


class MCPTool(Tool):
    """将 MCP Server 上的一个 tool 适配为 agentic 的 `Tool`。

    使用方式示例（伪代码）::

        client = YourMCPClientImplementation(...)
        mcp_tool = MCPTool(
            config=MCPToolConfig(
                server_name=\"my-mcp-server\",
                tool_name=\"search_docs\",
                description=\"Search internal documentation via MCP server\",
            ),
            client=client,
        )
        agent = Agent(..., tools=[mcp_tool])
    """

    def __init__(self, config: MCPToolConfig, client: MCPClient):
        self._config = config
        self._client = client

        # 尝试从 MCP server 拉 schema；失败则用本地默认 schema
        remote_schema = client.get_tool_schema(config.server_name, config.tool_name)
        self._parameters_schema = remote_schema or config.parameters_schema

    # ---- Tool 接口实现 -------------------------------------------------

    @property
    def name(self) -> str:
        # 在 agentic 内部，直接使用 MCP tool 的名字
        return self._config.tool_name

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def schema(self) -> ToolSchema:
        # 使用我们自己管理的 parameters_schema
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self._parameters_schema,
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        # 不再走基类默认逻辑，统一从构造时确定的 schema 读取
        return self._parameters_schema

    def execute(self, **kwargs: Any) -> Any:
        """通过 MCPClient 调用远端 MCP tool。"""
        return self._client.call_tool(
            server_name=self._config.server_name,
            tool_name=self._config.tool_name,
            arguments=kwargs,
            auth=self._config.auth,
        )

    def stream(self, **kwargs: Any) -> Iterable[Any]:
        """通过 MCPClient 以流式方式调用远端 MCP tool（如果实现了 stream_tool）。"""
        return self._client.stream_tool(
            server_name=self._config.server_name,
            tool_name=self._config.tool_name,
            arguments=kwargs,
            auth=self._config.auth,
        )

    def __repr__(self) -> str:
        return f"<MCPTool: {self._config.server_name}.{self._config.tool_name}>"

