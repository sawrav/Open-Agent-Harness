import httpx

from tools.base import BaseTool

MAX_RESPONSE_CHARS = 8000
TIMEOUT_SECONDS = 15


class FetchURLTool(BaseTool):
    name = "fetch_url"
    description = (
        "Fetch the text content of a URL and return it. "
        f"Response is capped at {MAX_RESPONSE_CHARS} characters."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch (must start with http:// or https://).",
            }
        },
        "required": ["url"],
    }

    async def run(self, url: str) -> str:
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": "OpenAgentHarness/1.0"},
                timeout=TIMEOUT_SECONDS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text[:MAX_RESPONSE_CHARS]
        except httpx.HTTPStatusError as e:
            return f"HTTP Error {e.response.status_code}: {e.response.reason_phrase}"
        except httpx.RequestError as e:
            return f"Request Error: {e}"
        except Exception as e:
            return f"Error: {e}"
