import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool
from langgraph.types import interrupt

from dotenv import load_dotenv
load_dotenv()

@tool
def send_email_tool(recipient_email: str, subject: str, content: str) -> str:
    """
    Sends an email using the system's common assistant Gmail account via SMTP.
    This function is dynamically executed by the LangGraph agent when ready.

    Args:
        recipient_email: The exact email address of the receiver.
        subject: The finalized subject line for the email.
        content: The COMPLETE body string of the email. This MUST include all 
                 paragraphs, line breaks, greetings, and the final signature 
                 (e.g., 'Regards, Shanks') exactly as approved by the human.
    """
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    sender_email = os.getenv("SMTP_USERNAME")
    sender_password = os.getenv("SMTP_PASSWORD")

    if not all([sender_email, sender_password]):
        return "Error: Missing assistant email credentials in configuration."

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(content, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade connection to secure TLS encryption
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()
        return f"Success: System assistant email successfully sent to {recipient_email}."
    except Exception as e:
        return f"SMTP Failure: Could not send email. Reason: {str(e)}"
    

@tool
def human_clarification(query: str) -> str:
    """ Use this tool to present a completed email draft to a human for formal 
    review, approval, or modification recommendations before sending."""
    reponse = interrupt({"query":query})
    return reponse["data"]
