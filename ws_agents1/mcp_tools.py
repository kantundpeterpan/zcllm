import json

class MCPTools:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.tools = None
    
    def get_tools(self):
        if self.tools is None:
            mcp_tools = self.mcp_client.get_tools()
            self.tools = convert_tools_list(mcp_tools)
        return self.tools

    def function_call(self, tool_call_response):
        function_name = tool_call_response.function.name
        arguments = json.loads(tool_call_response.function.arguments)
        call_id = tool_call_response.id

        result = self.mcp_client.call_tool(function_name, arguments)

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name":function_name,
            "content": json.dumps(
                search_results, indent=2
            )
        }
        # {
        #     "role": "function_call_output",
        #     "call_id": tool_call_response.call_id,
        #     "output": json.dumps(result, indent=2),
        # }
        