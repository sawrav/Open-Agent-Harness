"""
main.py — Entry point for OpenAgentHarness with vLLM backend.

Uses continuous batching and streaming callbacks for sympathetic agent harness.
"""

import asyncio
from agent import Agent


async def main():
    agent = Agent()
    try:
        await agent.chat()
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
