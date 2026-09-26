import re

import httpx

from tools.base import BaseTool

MAX_RESPONSE_CHARS = 3000   # ~750 tokens — keeps tool results from flooding context
TIMEOUT_SECONDS = 15


def _strip_html(html: str) -> str:
    """
    Strip HTML tags and collapse whitespace to extract readable text.
    Not a full parser — good enough for docs and README pages.
    """
    # Remove <script> and <style> blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") \
               .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace and blank lines
    lines = [line.strip() for line in html.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


def _rewrite_url(url: str) -> str:
    """
    Rewrite GitHub HTML URLs to their raw.githubusercontent equivalents
    so we get plain text instead of HTML.

    github.com/owner/repo/blob/branch/path  →  raw.githubusercontent.com/owner/repo/branch/path
    github.com/owner/repo  →  raw.githubusercontent.com/owner/repo/main/README.md
    """
    # blob URL → raw
    m = re.match(
        r"https?://github\.com/([^/]+/[^/]+)/blob/([^/]+)/(.*)",
        url,
    )
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"

    # bare repo URL → README
    m = re.match(r"https?://github\.com/([^/]+/[^/?#]+)/?$", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/main/README.md"

    return url


class FetchURLTool(BaseTool):
    name = "fetch_url"
    description = (
        "Fetch the text content of a URL and return clean readable text. "
        "HTML is stripped automatically. GitHub repo and blob URLs are "
        "redirected to raw content. "
        f"Response is capped at {MAX_RESPONSE_CHARS} characters (~750 tokens)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch (http:// or https://).",
            }
        },
        "required": ["url"],
    }

    async def run(self, url: str) -> str:
        # Rewrite GitHub HTML URLs to raw before fetching
        fetch_url = _rewrite_url(url)
        rewritten = fetch_url != url

        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": "OpenAgentHarness/1.0"},
                timeout=TIMEOUT_SECONDS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(fetch_url)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                text = resp.text

                # Strip HTML if response is HTML
                if "text/html" in content_type:
                    text = _strip_html(text)

                result = text[:MAX_RESPONSE_CHARS]
                if rewritten:
                    result = f"[Fetched raw content from: {fetch_url}]\n\n{result}"
                return result

        except httpx.HTTPStatusError as e:
            # If raw redirect 404s, fall back to original URL + HTML strip
            if rewritten and e.response.status_code == 404:
                try:
                    async with httpx.AsyncClient(
                        headers={"User-Agent": "OpenAgentHarness/1.0"},
                        timeout=TIMEOUT_SECONDS,
                        follow_redirects=True,
                    ) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        text = _strip_html(resp.text)
                        return text[:MAX_RESPONSE_CHARS]
                except Exception:
                    pass
            return f"HTTP Error {e.response.status_code}: {e.response.reason_phrase}"
        except httpx.RequestError as e:
            return f"Request Error: {e}"
        except Exception as e:
            return f"Error: {e}"
