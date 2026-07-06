import os
import sys  # <--- MAKE SURE THIS IS IMPORTED
from mcp.server.fastmcp import FastMCP
from slack_sdk import WebClient
import subprocess
import pathlib
import json

mcp = FastMCP()
@mcp.tool(
    name='CSV_reading_tool',
    description=(
        "A tool that reads CSV or Excel (.xlsx) files from the user's active directory folder. "
        "Args: filename: The raw name of the file (e.g., 'sales_data.csv' or 'data.xlsx')"
    )
)
def read_csv_data(filename: str) -> str:
    # DYNAMICALLY RESOLVE ACTIVE CONTEXT
    active_dir = pathlib.Path(os.getcwd())
    path = active_dir / filename
    try:
        # Check if it's an Excel file and parse it with the openpyxl engine we installed
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            import pandas as pd
            df = pd.read_excel(path, engine='openpyxl')
            return f"Excel file loaded successfully! Previewing the first 10 rows:\n\n{df.head(10).to_string()}"
            
        return path.read_text(encoding='utf-8')
    except Exception as e:
        available_files = [f.name for f in active_dir.iterdir() if f.is_file()]
        return f"Error opening the file at {path}: {e}. Files present in this execution path: {available_files}"

@mcp.tool(
    name='Execute_code_sandbox',
    description = (''' Safely executes dynamically written python data analysis
                    code locally and captures stdout, stderr and generated plots.
                     Args:
                         script_content: Full multi-line python code string to run.
''')
)
def execute_analysis_code(script_content: str) -> str:
    active_dir = pathlib.Path(os.getcwd())
    sandbox_file = active_dir / 'analysis_script.py'
    try:
        # Prepend a headless configuration so matplotlib doesn't throw GUI errors on users' machines
        headless_prefix = "import matplotlib\nmatplotlib.use('Agg')\n"
        sandbox_file.write_text(headless_prefix + script_content, encoding='utf-8')
        
        # FIXED: Removed quotes around sys.executable to pass the variable, not the string!
        result = subprocess.run([sys.executable, sandbox_file], capture_output=True, text=True, timeout=30)
        return f"STDOUT EXECUTED:\n{result.stdout}\nSTDERR EXECUTED:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return 'Process error: Script killed due to execution timeout of 30 seconds max'
    except Exception as e:
        return f'process error: {str(e)}'
    finally:
        if sandbox_file.exists():
            sandbox_file.unlink()  


@mcp.tool(
    name='Slack_message_tool',
    description=(''' 
    Dispatches markdown text analysis summaries along with data plots 
    directly to the company Slack workspace channel configured in the system files.
    
    Args:
        message_text: Summary text containing insights.
        plot_name: Optional file name of an image plot sitting inside the working directory.
 ''')
)
def send_slack_report(message_text: str, plot_name: str = None) -> str:
    # Resolve the shared configuration file location
    config_path = pathlib.Path.home() / ".mcp_analyst_config.json"
    
    if not config_path.exists():
        return "configuration error: No local config file found. Please initialize via cli.py first."
        
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
            token = config_data.get("SLACK_BOT_TOKEN")
            channel = config_data.get("SLACK_CHANNEL_ID")
    except Exception as e:
        return f"configuration error: Failed to parse configuration file: {str(e)}"

    # Dynamic Validation
    if not token:
        return "configuration error: SLACK_BOT_TOKEN is missing from your config profile."
    if not channel:
        return "configuration error: SLACK_CHANNEL_ID is missing from your config profile."
        
    client = WebClient(token=token)
    active_dir = pathlib.Path(os.getcwd())
    
    try:
        if plot_name:
            plot_path = active_dir / plot_name
            if not plot_path.exists():
                return f"error: plot file '{plot_name}' does not exist in the working directory ({active_dir})"
            
            client.files_upload_v2(
                channel=channel,
                file=str(plot_path),
                initial_comment=message_text
            )
            
        else:
            client.chat_postMessage(
                channel=channel,
                text=message_text
            )
        return f"Slack message sent successfully to configured channel: {channel}"
    except Exception as e:
        return f'slack API failure : {str(e)}'
        
if __name__ == '__main__':
    mcp.run()