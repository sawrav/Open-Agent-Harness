"""
Google Docs tools for reading and modifying Google Docs using OAuth with authorization token.

Tools:
- read_google_doc: Read content from a Google Doc
- update_google_doc: Update content in a Google Doc
"""

import asyncio
import base64
import httpx

from tools.base import BaseTool

MAX_RESPONSE_CHARS = 3000
TIMEOUT_SECONDS = 15


class GoogleDocsReadTool(BaseTool):
    name = "read_google_doc"
    description = (
        "Read content from a Google Doc using OAuth authorization token. "
        "Requires authorization token with Google Docs read permissions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "Google Doc ID (from URL: docs.google.com/document/d/{ID}/edit)",
            },
            "authorization_token": {
                "type": "string",
                "description": "OAuth2 authorization token for Google Docs API",
            },
        },
        "required": ["document_id", "authorization_token"],
    }

    async def run(self, document_id: str, authorization_token: str) -> str:
        """Read content from Google Doc."""
        url = f"https://docs.googleapis.com/v1/documents/{document_id}"
        headers = {
            "Authorization": f"Bearer {authorization_token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                headers=headers,
                timeout=TIMEOUT_SECONDS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                
                # Extract text content from document
                content_parts = []
                for element in data.get("body", {}).get("content", []):
                    paragraph = element.get("paragraph", {})
                    for element_data in paragraph.get("elements", []):
                        text_run = element_data.get("textRun", {})
                        if text_run and "content" in text_run:
                            content_parts.append(text_run["content"])
                
                content = "".join(content_parts)
                if len(content) > MAX_RESPONSE_CHARS:
                    content = content[:MAX_RESPONSE_CHARS] + f"\n... [truncated, {len(content)} total characters]"
                
                return content if content else "Document is empty"
                
        except httpx.HTTPStatusError as e:
            return f"HTTP Error {e.response.status_code}: {e.response.reason_phrase}"
        except httpx.RequestError as e:
            return f"Request Error: {e}"
        except Exception as e:
            return f"Error reading Google Doc: {e}"


class GoogleDocsWriteTool(BaseTool):
    name = "update_google_doc"
    description = (
        "Update content in a Google Doc using OAuth authorization token. "
        "Requires authorization token with Google Docs write permissions. "
        "Appends content at the end of the document."
    )
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "Google Doc ID (from URL: docs.google.com/document/d/{ID}/edit)",
            },
            "authorization_token": {
                "type": "string",
                "description": "OAuth2 authorization token for Google Docs API",
            },
            "content": {
                "type": "string",
                "description": "Content to append to the document",
            },
            "encoding": {
                "type": "string",
                "enum": ["utf-8", "base64"],
                "description": "Encoding of content. Use base64 for code/special characters",
            },
        },
        "required": ["document_id", "authorization_token", "content"],
    }

    async def run(self, document_id: str, authorization_token: str, content: str, encoding: str = "utf-8") -> str:
        """Update content in Google Doc."""
        # Decode content if base64 encoded
        if encoding == "base64":
            try:
                raw_bytes = base64.b64decode(content)
                decoded = raw_bytes.decode("utf-8")
            except Exception as e:
                return f"Error: could not decode base64 content: {e}"
            write_content = decoded
        else:
            write_content = content
        
        url = f"https://docs.googleapis.com/v1/documents/{document_id}:batchUpdate"
        headers = {
            "Authorization": f"Bearer {authorization_token}",
            "Content-Type": "application/json",
        }
        
        try:
            async with httpx.AsyncClient(
                headers=headers,
                timeout=TIMEOUT_SECONDS,
            ) as client:
                # Get current document to find end index
                get_resp = await client.get(f"https://docs.googleapis.com/v1/documents/{document_id}")
                get_resp.raise_for_status()
                doc_data = get_resp.json()
                
                # Find end index (approximate)
                end_index = 1
                for element in doc_data.get("body", {}).get("content", []):
                    end_index = element.get("endIndex", end_index)
                
                # Prepare batch update request to insert at end
                payload = {
                    "requests": [{
                        "insertText": {
                            "location": {"index": max(1, end_index - 1)},
                            "text": write_content
                        }
                    }]
                }
                
                update_resp = await client.post(url, json=payload)
                update_resp.raise_for_status()
                
                return f"Successfully appended {len(write_content)} characters to document {document_id}"
                
        except httpx.HTTPStatusError as e:
            return f"HTTP Error {e.response.status_code}: {e.response.reason_phrase}"
        except httpx.RequestError as e:
            return f"Request Error: {e}"
        except Exception as e:
            return f"Error updating Google Doc: {e}"
