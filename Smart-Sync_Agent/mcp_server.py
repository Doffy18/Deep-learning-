from mcp.server.fastmcp import FastMCP
from pydantic import Field
import os
from pathlib import Path
from typing import Literal, Optional
from mcp.server.fastmcp import FastMCP, Context
import httpx


jina_api_key=os.getenv("JINA_API_KEY")
DEFAULT_TEST_ROOT = Path(os.getenv("M_WORKSPACE_ROOT", "./test_notes")).resolve()


mcp = FastMCP('CliMCP', log_level='ERROR')




def file_url_to_path(uri: str) -> Path:
    """Helper to convert standard file:// URIs from client roots into clear system Paths."""
    if uri.startswith("file://"):
        return Path(uri[7:]).resolve()
    return Path(uri).resolve()



async def is_path_allowed(requested_path: Path, ctx: Context) -> bool:
    """
    Validates if a target path falls safely inside any of the 
    dynamically defined client roots using native MCP context.
    """
    try:
        # Try fetching roots from client session context
        roots_result = await ctx.session.list_roots()
        client_roots = roots_result.roots
    except Exception:
        # Fallback if client doesn't support the list_roots capability
        client_roots = []

    # FALLBACK FOR TESTING / NON-SUPPORTED CLIENTS
    if not client_roots:
        try:
            requested_path.parent.resolve().relative_to(DEFAULT_TEST_ROOT)
            return True
        except ValueError:
            return False

    # Resolve target directory level
    target_dir = requested_path.parent if requested_path.suffix else requested_path
    target_dir = target_dir.resolve()

    for root in client_roots:
        root_path = file_url_to_path(root.uri)
        try:
            target_dir.relative_to(root_path)
            return True
        except ValueError:
            continue

    return False





@mcp.tool(
    name='local_note_manager',
    description=(
        'A secure, unified file manager to manipulate local markdown notes. '
        'Allows reading, writing, appending, and listing directory files. '
        'All path activities are restricted within the permitted workspace roots.'
    )
)

async def local_note_manager(
    action: Literal["read", "write", "append", "list_dir"] = Field(
        description="The physical file operation to perform."
    ),
    filename: str = Field(
        default="index.md",
        description="The filename or path string relative to the workspace root (e.g., 'index.md' or 'databases/postgres.md')."
    ),
    content: Optional[str] = Field(
        default=None,
        description="The structured markdown string body payload to insert or append. Required for 'write' and 'append' actions."
    ),
    *,
    ctx: Context
) -> str:
   
    try:
        roots_result = await ctx.session.list_roots()
        if not roots_result.roots:
            base_root = DEFAULT_TEST_ROOT
        else:
            base_root = file_url_to_path(roots_result.roots[0].uri)
    except Exception:
        # Fallback entirely if client raises "List roots not supported"
        base_root = DEFAULT_TEST_ROOT
    
    requested_path = Path(base_root / filename).resolve()
    
    # 2. Security validation check
    if not await is_path_allowed(requested_path, ctx):
        return f"Permission Denied: Target path layout configuration for '{filename}' violates secure root boundaries."

    # 3. Action routing
    try:
        # ACTION: READ NOTE
        if action == "read":
            if not requested_path.exists() or not requested_path.is_file():
                return f"Error: Note file '{filename}' does not exist."
            data = requested_path.read_text(encoding='utf-8')
            return f"--- START OF FILE: {filename} ---\n{data}\n--- END OF FILE ---"

        # ACTION: WRITE/OVERWRITE NOTE
        elif action == "write":
            if content is None:
                return "Error: Valid string 'content' field payload is required for write operation."
            
            # Dynamically create subdirectories if they don't exist yet
            requested_path.parent.mkdir(parents=True, exist_ok=True)
            requested_path.write_text(content, encoding="utf-8")
            return f"Success: Safely written payload to note path '{filename}'."

        # ACTION: APPEND TO EXISTING NOTE
        elif action == "append":
            if content is None:
                return "Error: Valid string 'content' field payload is required for append operation."
            
            requested_path.parent.mkdir(parents=True, exist_ok=True)
            # Use append mode with a clean newline break
            with open(requested_path, "a", encoding="utf-8") as file_handle:
                file_handle.write(f"\n{content}")
            return f"Success: Content successfully appended to target file '{filename}'."

        # ACTION: LIST WORKSPACE FILES
        elif action == "list_dir":
            # If the user targets the root, use base_root, otherwise check specified subdirectory
            target_directory = requested_path if requested_path.is_dir() else requested_path.parent
            if not target_directory.exists():
                return f"Error: Directory layout structure for '{filename}' does not exist."
            
            # Scan files inside the verified folder
            entries = [item.name for item in target_directory.iterdir() if not item.name.startswith('.')]
            return f"Directory files found within path '{filename}':\n" + "\n".join([f"- {name}" for name in entries])

    except Exception as error_exception:
        return f"System Runtime Error processing operation: {str(error_exception)}"
    


@mcp.tool(
    name='web_content_extractor',
    description=(
        'Fetches an external URL and extracts its core contents as clean, stripped, '
        'human-readable Markdown. Removes navigation headers, ads, and tracking scripts.'
    )
)
async def web_content_extractor(
    url: str = Field(description="The full HTTP/HTTPS URL of the web page or article to scrape.")
) -> str:
    """
    Interfaces directly with the Jina AI Reader API to convert 
    raw HTML pages into pristine Markdown strings.
    """
    # Target URL format required by Jina Reader API
    jina_endpoint = f"https://r.jina.ai/{url}"
    
    # Check if a Jina API key is available in your environment variables
    # (Optional for free tier, but raises rate-limits if present)
    headers = {"Accept": "text/markdown"}
    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"

    try:
        # Use an async HTTP client to avoid blocking the server loop
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(jina_endpoint, headers=headers)
            
            if response.status_code != 200:
                return f"Failed to extract web content. Jina API returned status code: {response.status_code}"
                
            return response.text

    except httpx.RequestError as exc:
        return f"An error occurred while connecting to the scraper backend: {str(exc)}"
    

if __name__ == "__main__":
        mcp.run()