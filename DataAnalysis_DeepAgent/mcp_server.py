import os

from mcp.server.fastmcp import FastMCP
from slack_sdk import WebClient
import subprocess
import pathlib

mcp  = FastMCP()


DATA_dir = pathlib.Path.cwd() / "data"
DATA_dir.mkdir(parents=True, exist_ok=True)

@mcp.tool(
    name='CSV_reading_tool',
    description = ( '''
    A tool that reads CSV files from the sandboxed data directory.
    Args:
        file_name: The raw name of the files(e.g., 'sales_data.csv)
    '''
    )
)
def read_csv_data(filename:str)-> str:
    path = pathlib.Path(DATA_dir) / filename
    try:
        return path.read_text(encoding='utf-8')
    except Exception as e:
        return f'error reading the csv file: {e}'




@mcp.tool(
    name= 'Execute_code_sandbox',
    descrition = ('''  Safetly executes dynamically written python data analysis
                   code locally and captures stdout, stderr and generated plots.
                    Args:
                        script_content: Full multi-line python code string to run.
''')
)
def execute_analysis_code(script_content:str)-> str:
    sandbox_file = pathlib.Path(DATA_dir) / 'analysis_script.py'
    try:
        sandbox_file.write_text(script_content, encoding='utf-8')
        result = subprocess.run(['python', sandbox_file], capture_output=True, text=True, timeout=30)
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
    description = (''' 
    Dispatches markdown text analysis summaries along with data plots 
    to a company Slack workspace channel.
    
    Args:
        channel: The targeted Slack channel ID or name string (e.g., 'C0123456ABC').
        message_text: Summary text containing insights.
        plot_name: Optional file name of an image plot sitting inside the data directory.
 ''')
)
def send_slack_report(channel: str, message_text: str, plot_name: str = None) -> str:
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        return "configuration error: slack_bot_token is not set on the target host server"
    client = WebClient(token=token)
    try:
        if plot_name:
            plot_path = pathlib.Path(DATA_dir) / plot_name
            if not plot_path.exists():
                return f"error: plot file '{plot_name}' does not exist in the data directory"
            
            client.files_upload_v2(
                channel=channel,
                file=plot_path,
                initial_comment=message_text
            )
            
        else:
            client.chat_postMessage(
                channel=channel,
                text=message_text
            )
        return "Slack message sent successfully"
    except Exception as e:
        return f'stack API failure : {str(e)}'
        
if __name__ == '__main__':
    mcp.run()