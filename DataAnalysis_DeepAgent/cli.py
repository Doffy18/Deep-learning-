import os
import json
import asyncio
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from deep_agent import main as run_agent_workflow

app = typer.Typer(help="MCP Autonomous Data Analyst Space Console Tool.")
console = Console()

CONFIG_FILE = Path.home() / ".mcp_analyst_config.json"

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(gemini_key: str, slack_token: str):
    config_data = {
        "GEMINI_API_KEY": gemini_key,
        "SLACK_BOT_TOKEN": slack_token
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        console.print(f"[yellow]⚠️ Warning: Could not cache config locally: {e}[/yellow]")

def run_interactive_prompts(cached_config: dict) -> tuple:
    console.print("\n[bold yellow]⚙️ Please fill or verify your structural access credentials:[/bold yellow]")
    gemini_key = typer.prompt("🔑 Enter your Gemini API Key", hide_input=True, show_default=False)
    slack_token = typer.prompt("🔑 Enter your Slack Bot User Token (xoxb-...)", hide_input=True, show_default=False)
    return gemini_key, slack_token

@app.callback(invoke_without_command=True)
def main():
    console.print(Panel("[bold gold1]📊 MCP Data Analyst Workspace Context Initialization[/bold gold1]", border_style="cyan"))

    cached_config = load_config()
    gemini_key = cached_config.get("GEMINI_API_KEY")
    slack_token = cached_config.get("SLACK_BOT_TOKEN")

    if not gemini_key or not slack_token:
        gemini_key, slack_token = run_interactive_prompts(cached_config)

    # Export configuration straight into process states
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["SLACK_BOT_TOKEN"] = slack_token
    save_config(gemini_key, slack_token)

    console.print("\n[bold green]✔ Verification Complete. Session Connected Securely![/bold green]")
    console.print(Panel(
        f"[bold magenta]Current Active Workspace Platform Directory Context:[/bold magenta]\n"
        f"📂 {Path.cwd().resolve()}\n\n"
        f"[bold yellow]Available Configurations & Resets Short-codes:[/bold yellow]\n"
        f"  • [cyan]:config[/cyan]       -> Reset/Update all configuration keys\n"
        f"  • [cyan]:config:gemini[/cyan] -> Update Gemini API Key credentials exclusively\n"
        f"  • [cyan]:config:slack[/cyan]  -> Update Slack Bot Token credentials exclusively\n"
        f"  • [cyan]exit[/cyan] / [cyan]quit[/cyan]   -> Terminate current process loop cleanly",
        border_style="magenta"
    ))

    while True:
        try:
            user_input = console.input("[bold cyan]analyst-mcp ❯ [/bold cyan]").strip()

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold red]Closing terminal loop connection window context. Goodbye![/bold red]\n")
                break

            if not user_input:
                continue

            # --- Target Specific Configuration Resets ---
            if user_input.lower().startswith(":config"):
                cmd = user_input.lower()
                cached_config = load_config()

                if cmd == ":config:gemini":
                    gemini_key = typer.prompt("🔑 Enter your NEW Gemini API Key", hide_input=True, show_default=False)
                    os.environ["GEMINI_API_KEY"] = gemini_key
                    save_config(gemini_key, os.environ.get("SLACK_BOT_TOKEN", ""))
                    console.print("[bold green]✔ Gemini API Key reset successfully![/bold green]\n")
                elif cmd == ":config:slack":
                    slack_token = typer.prompt("🔑 Enter your NEW Slack Bot User Token", hide_input=True, show_default=False)
                    os.environ["SLACK_BOT_TOKEN"] = slack_token
                    save_config(os.environ.get("GEMINI_API_KEY", ""), slack_token)
                    console.print("[bold green]✔ Slack Bot Token reset successfully![/bold green]\n")
                elif cmd == ":config":
                    gemini_key, slack_token = run_interactive_prompts(cached_config)
                    os.environ["GEMINI_API_KEY"] = gemini_key
                    os.environ["SLACK_BOT_TOKEN"] = slack_token
                    save_config(gemini_key, slack_token)
                    console.print("[bold green]✔ All configuration settings updated successfully![/bold green]\n")
                else:
                    console.print("[bold red]❌ Unknown target pattern. Use :config, :config:gemini, or :config:slack[/bold red]\n")
                continue

            with Status("[bold green]Agent is running analytical execution loops...[/bold green]", spinner="dots"):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                agent_feedback = loop.run_until_complete(
                    run_agent_workflow(user_input_text=user_input)
                )

            console.print(f"\n[bold gold1]🤖 Analyst Execution Response Output:[/bold gold1]\n{agent_feedback}\n")

        except KeyboardInterrupt:
            console.print("\n[bold red]Session closed cleanly via break flag signal.[/bold red]\n")
            break
        except Exception as e:
            console.print(f"[bold red]System Exception triggered:[/bold red] {e}\n")

if __name__ == "__main__":
    app()