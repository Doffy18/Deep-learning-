import os
import json
import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from agent import run_agent_workflow

app = typer.Typer(help="Smart-Sync CLI Interactive Knowledge Base Space.")
console = Console()

# Hidden configuration storage path in the user's home directory
CONFIG_FILE = Path.home() / ".smart_sync_config.json"

def load_config() -> dict:
    """Reads cached configuration if it exists."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(gemini_key: str, jina_key: str, root_dir: str):
    """Caches configurations to a local hidden JSON file."""
    config_data = {
        "GEMINI_API_KEY": gemini_key,
        "JINA_API_KEY": jina_key,
        "M_WORKSPACE_ROOT": root_dir
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        console.print(f"[yellow]⚠️ Warning: Could not cache configuration file locally: {e}[/yellow]")

def normalize_windows_path(path_str: str) -> str:
    """
    Safely intercepts Git Bash style Unix pathways (e.g. '/c/Users/...')
    and normalizes them into pristine Windows filesystem strings.
    """
    if path_str.startswith('/') and len(path_str) > 2 and path_str[2] == '/':
        drive_letter = path_str[1].upper()
        rest_of_path = path_str[2:].replace('/', '\\')
        path_str = f"{drive_letter}:{rest_of_path}"
    return str(Path(path_str).resolve())

def run_interactive_prompts(cached_config: dict) -> tuple:
    """Prompts the user for all keys sequentially."""
    console.print("\n[bold yellow]⚙️ Please update or verify your configurations:[/bold yellow]")
    default_root = cached_config.get("M_WORKSPACE_ROOT", "./test_notes")
    
    gemini_key = typer.prompt("🔑 Enter your Gemini API Key", hide_input=True, show_default=False)
    jina_key = typer.prompt("🔑 Enter your Jina AI API Key (Press Enter to use Free Tier)", default="", hide_input=True, show_default=False)
    root_dir = typer.prompt("📂 Enter your Local Workspace Root Pathway", default=default_root)
    
    return gemini_key, jina_key, root_dir

@app.callback(invoke_without_command=True)
def main():
    """
    Launches the primary interactive session loop for Smart-Sync directly.
    """
    console.print(Panel("[bold gold1]🤖 Smart-Sync Workspace Configuration Initialization[/bold gold1]", border_style="cyan"))

    # Load existing configurations from cache
    cached_config = load_config()
    
    gemini_key = cached_config.get("GEMINI_API_KEY")
    jina_key = cached_config.get("JINA_API_KEY")
    root_dir = cached_config.get("M_WORKSPACE_ROOT")

    # Setup triggers if any necessary configuration value is missing entirely
    if not gemini_key or jina_key is None or not root_dir:
        gemini_key, jina_key, root_dir = run_interactive_prompts(cached_config)

    # Normalize pathway schema targeting cross-platform reliability
    resolved_root_path = normalize_windows_path(root_dir)

    # Set initial environment updates
    os.environ["JINA_API_KEY"] = jina_key
    Path(resolved_root_path).mkdir(parents=True, exist_ok=True)
    
    # Save the cleaned path back to file structure cache
    save_config(gemini_key, jina_key, resolved_root_path)

    # Enter the persistent chat session
    console.print("\n[bold green]✔ Environment Context successfully verified![/bold green]")
    console.print(Panel(
        "[bold magenta]🤖 Welcome to your Smart-Sync Interactive Session Space.[/bold magenta]\n"
        "Ask general questions, or issue file manipulation tasks natively.\n\n"
        "[bold yellow]Configuration Shortcuts Available Anywhere:[/bold yellow]\n"
        "  • [cyan]:config[/cyan]            -> Reset all settings sequentially\n"
        "  • [cyan]:config:gemini[/cyan]     -> Update the Gemini API Key only\n"
        "  • [cyan]:config:jina[/cyan]       -> Update the Jina AI API Key only\n"
        "  • [cyan]:config:workspace[/cyan]  -> Update the Workspace Folder Path only\n"
        "  • [cyan]exit[/cyan] / [cyan]quit[/cyan]       -> Close out connection cleanly",
        border_style="magenta"
    ))

    # Continuous chat loop execution
    while True:
        try:
            user_input = console.input("[bold cyan]smart-sync ❯ [/bold cyan]").strip()

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold red]Exiting session layout context. Goodbye![/bold red]\n")
                break

            if not user_input:
                continue

            # =====================================================================
            # CONFIGURATION SHORTCUT INTERCEPTION ROUTER
            # =====================================================================
            if user_input.lower().startswith(":config"):
                cmd = user_input.lower()
                cached_config = load_config()
                
                if cmd == ":config:gemini":
                    gemini_key = typer.prompt("🔑 Enter your Gemini API Key", hide_input=True, show_default=False)
                    console.print("[bold green]✔ Gemini API Key updated locally![/bold green]\n")
                elif cmd == ":config:jina":
                    jina_key = typer.prompt("🔑 Enter your Jina AI API Key (Press Enter to use Free Tier)", default="", hide_input=True, show_default=False)
                    os.environ["JINA_API_KEY"] = jina_key
                    console.print("[bold green]✔ Jina AI Key updated locally![/bold green]\n")
                elif cmd == ":config:workspace":
                    default_root = cached_config.get("M_WORKSPACE_ROOT", "./test_notes")
                    root_dir = typer.prompt("📂 Enter your Local Workspace Root Pathway", default=default_root)
                    resolved_root_path = normalize_windows_path(root_dir)
                    Path(resolved_root_path).mkdir(parents=True, exist_ok=True)
                    console.print("[bold green]✔ Workspace path resolved and updated locally![/bold green]\n")
                elif cmd == ":config":
                    gemini_key, jina_key, root_dir = run_interactive_prompts(cached_config)
                    os.environ["JINA_API_KEY"] = jina_key
                    resolved_root_path = normalize_windows_path(root_dir)
                    Path(resolved_root_path).mkdir(parents=True, exist_ok=True)
                    console.print("[bold green]✔ Complete context configuration updated successfully![/bold green]\n")
                else:
                    console.print("[bold red]❌ Unknown shortcut option. Use :config, :config:gemini, :config:jina, or :config:workspace[/bold red]\n")
                    continue
                
                # Save the new values back to file structure cache
                save_config(gemini_key, jina_key, resolved_root_path)
                continue

            # Run LangGraph workflow execution loop
            with Status("[bold green]Agent is thinking...[/bold green]", spinner="dots"):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                agent_feedback = loop.run_until_complete(
                    run_agent_workflow(
                        user_input_text=user_input, 
                        gemini_key=gemini_key, 
                        root_dir=resolved_root_path
                    )
                )

            console.print(f"\n[bold gold1]🤖 Smart-Sync Response:[/bold gold1]\n{agent_feedback}\n")

        except KeyboardInterrupt:
            console.print("\n[bold red]Session closed cleanly via break flag signal.[/bold red]\n")
            break
        except Exception as e:
            def print_exception_deeply(err, depth=1):
                if hasattr(err, "exceptions") and err.exceptions:
                    for sub_err in err.exceptions:
                        print_exception_deeply(sub_err, depth + 1)
                else:
                    console.print(f"{'  ' * depth}[bold red]💥 Root Cause Exception:[/bold red] [yellow]{type(err).__name__}[/yellow]: {err}")

            console.print(f"[bold red]System Runtime Exception error triggered (TaskGroup Stack):[/bold red]")
            print_exception_deeply(e)
            console.print("") 

if __name__ == "__main__":
    app()


# # cli.py
# import os
# import json
# import asyncio
# from pathlib import Path

# import typer
# from rich.console import Console
# from rich.panel import Panel
# from rich.status import Status

# from agent import run_agent_workflow

# app = typer.Typer(help="Smart-Sync CLI Interactive Knowledge Base Space.")
# console = Console()

# # Hidden configuration storage path in the user's home directory
# CONFIG_FILE = Path.home() / ".smart_sync_config.json"

# def load_config() -> dict:
#     """Reads cached configuration if it exists."""
#     if CONFIG_FILE.exists():
#         try:
#             with open(CONFIG_FILE, "r") as f:
#                 return json.load(f)
#         except Exception:
#             return {}
#     return {}

# def save_config(gemini_key: str, jina_key: str, root_dir: str):
#     """Caches configurations to a local hidden JSON file."""
#     config_data = {
#         "GEMINI_API_KEY": gemini_key,
#         "JINA_API_KEY": jina_key,
#         "M_WORKSPACE_ROOT": root_dir
#     }
#     try:
#         with open(CONFIG_FILE, "w") as f:
#             json.dump(config_data, f, indent=4)
#     except Exception as e:
#         console.print(f"[yellow]⚠️ Warning: Could not cache configuration file locally: {e}[/yellow]")

# def run_interactive_prompts(cached_config: dict) -> tuple:
#     """Prompts the user for all keys sequentially."""
#     console.print("\n[bold yellow]⚙️ Please update or verify your configurations:[/bold yellow]")
#     default_root = cached_config.get("M_WORKSPACE_ROOT", "./test_notes")
    
#     gemini_key = typer.prompt("🔑 Enter your Gemini API Key", hide_input=True, show_default=False)
#     jina_key = typer.prompt("🔑 Enter your Jina AI API Key (Press Enter to use Free Tier)", default="", hide_input=True, show_default=False)
#     root_dir = typer.prompt("📂 Enter your Local Workspace Root Pathway", default=default_root)
    
#     return gemini_key, jina_key, root_dir

# @app.callback(invoke_without_command=True)
# def main():
#     """
#     Launches the primary interactive session loop for Smart-Sync directly.
#     """
#     console.print(Panel("[bold gold1]🤖 Smart-Sync Workspace Configuration Initialization[/bold gold1]", border_style="cyan"))

#     # Load existing configurations from cache
#     cached_config = load_config()
    
#     gemini_key = cached_config.get("GEMINI_API_KEY")
#     jina_key = cached_config.get("JINA_API_KEY")
#     root_dir = cached_config.get("M_WORKSPACE_ROOT")

#     # Setup triggers if any necessary configuration value is missing entirely
#     if not gemini_key or jina_key is None or not root_dir:
#         gemini_key, jina_key, root_dir = run_interactive_prompts(cached_config)
#         save_config(gemini_key, jina_key, root_dir)

#     # Set initial environment updates
#     os.environ["JINA_API_KEY"] = jina_key
#     resolved_root_path = str(Path(root_dir).resolve())
#     Path(resolved_root_path).mkdir(parents=True, exist_ok=True)

#     # Enter the persistent chat session
#     console.print("\n[bold green]✔ Environment Context successfully verified![/bold green]")
#     console.print(Panel(
#         "[bold magenta]🤖 Welcome to your Smart-Sync Interactive Session Space.[/bold magenta]\n"
#         "Ask general questions, or issue file manipulation tasks natively.\n\n"
#         "[bold yellow]Configuration Shortcuts Available Anywhere:[/bold yellow]\n"
#         "  • [cyan]:config[/cyan]           -> Reset all settings sequentially\n"
#         "  • [cyan]:config:gemini[/cyan]     -> Update the Gemini API Key only\n"
#         "  • [cyan]:config:jina[/cyan]       -> Update the Jina AI API Key only\n"
#         "  • [cyan]:config:workspace[/cyan]  -> Update the Workspace Folder Path only\n"
#         "  • [cyan]exit[/cyan] / [cyan]quit[/cyan]       -> Close out connection cleanly",
#         border_style="magenta"
#     ))

#     # Continuous chat loop execution
#     while True:
#         try:
#             user_input = console.input("[bold cyan]smart-sync ❯ [/bold cyan]").strip()

#             if user_input.lower() in ["exit", "quit"]:
#                 console.print("[bold red]Exiting session layout context. Goodbye![/bold red]\n")
#                 break

#             if not user_input:
#                 continue

#             # =====================================================================
#             # CONFIGURATION SHORTCUT INTERCEPTION ROUTER
#             # =====================================================================
#             if user_input.lower().startswith(":config"):
#                 cmd = user_input.lower()
#                 cached_config = load_config()
                
#                 if cmd == ":config:gemini":
#                     gemini_key = typer.prompt("🔑 Enter your Gemini API Key", hide_input=True, show_default=False)
#                     console.print("[bold green]✔ Gemini API Key updated locally![/bold green]\n")
#                 elif cmd == ":config:jina":
#                     jina_key = typer.prompt("🔑 Enter your Jina AI API Key (Press Enter to use Free Tier)", default="", hide_input=True, show_default=False)
#                     os.environ["JINA_API_KEY"] = jina_key
#                     console.print("[bold green]✔ Jina AI Key updated locally![/bold green]\n")
#                 elif cmd == ":config:workspace":
#                     default_root = cached_config.get("M_WORKSPACE_ROOT", "./test_notes")
#                     root_dir = typer.prompt("📂 Enter your Local Workspace Root Pathway", default=default_root)
#                     resolved_root_path = str(Path(root_dir).resolve())
#                     Path(resolved_root_path).mkdir(parents=True, exist_ok=True)
#                     console.print("[bold green]✔ Workspace path resolved and updated locally![/bold green]\n")
#                 elif cmd == ":config":
#                     # Fallback to updating everything at once sequentially
#                     gemini_key, jina_key, root_dir = run_interactive_prompts(cached_config)
#                     os.environ["JINA_API_KEY"] = jina_key
#                     resolved_root_path = str(Path(root_dir).resolve())
#                     Path(resolved_root_path).mkdir(parents=True, exist_ok=True)
#                     console.print("[bold green]✔ Complete context configuration updated successfully![/bold green]\n")
#                 else:
#                     console.print("[bold red]❌ Unknown shortcut option. Use :config, :config:gemini, :config:jina, or :config:workspace[/bold red]\n")
#                     continue
                
#                 # Save the new values back to file structure cache
#                 save_config(gemini_key, jina_key, root_dir)
#                 continue

#             # Run LangGraph workflow execution loop
#             with Status("[bold green]Agent is thinking...[/bold green]", spinner="dots"):
#                 try:
#                     loop = asyncio.get_running_loop()
#                 except RuntimeError:
#                     loop = asyncio.new_event_loop()
#                     asyncio.set_event_loop(loop)
                    
#                 agent_feedback = loop.run_until_complete(
#                     run_agent_workflow(
#                         user_input_text=user_input, 
#                         gemini_key=gemini_key, 
#                         root_dir=resolved_root_path
#                     )
#                 )

#             console.print(f"\n[bold gold1]🤖 Smart-Sync Response:[/bold gold1]\n{agent_feedback}\n")

#         except KeyboardInterrupt:
#             console.print("\n[bold red]Session closed cleanly via break flag signal.[/bold red]\n")
#             break
#         except Exception as e:
#             def print_exception_deeply(err, depth=1):
#                 # If it's an ExceptionGroup, dig into its sub-exceptions
#                 if hasattr(err, "exceptions") and err.exceptions:
#                     for sub_err in err.exceptions:
#                         print_exception_deeply(sub_err, depth + 1)
#                 else:
#                     # We reached the actual root cause!
#                     console.print(f"{'  ' * depth}[bold red]💥 Root Cause Exception:[/bold red] [yellow]{type(err).__name__}[/yellow]: {err}")

#             console.print(f"[bold red]System Runtime Exception error triggered (TaskGroup Stack):[/bold red]")
#             print_exception_deeply(e)
#             console.print("") # Blank line for spacing

# if __name__ == "__main__":
#     app()
