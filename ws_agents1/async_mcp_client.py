from fastmcp import Client
import asyncio

async def main():
    async with Client("./weather_server.py") as client:
         # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        print(tools)
        
if __name__ == "__main__":
    asyncio.run(main())