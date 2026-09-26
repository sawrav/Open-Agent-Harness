import urllib.request
import urllib.error
from tools.base import BaseTool

# Maximum characters returned from a fetched page to avoid flooding the context.
MAX_RESPONSE_CHARS = 8000


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

    def run(self, url: str) -> str:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "OpenAgentHarness/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode(errors="replace")
                return content[:MAX_RESPONSE_CHARS]
        except urllib.error.HTTPError as e:
            return f"HTTP Error {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            return f"URL Error: {e.reason}"
        except Exception as e:
            return f"Error: {e}"
