"""Optional SMTP notifications for completed runs; credentials come from the environment."""

import argparse
from contextlib import suppress
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import json
import os
from pathlib import Path
import re
import smtplib
import ssl
import sys
from zoneinfo import ZoneInfo


def _send(record, recipient, env):
    required = ("KUANG_SMTP_HOST", "KUANG_SMTP_USER", "KUANG_SMTP_PASSWORD")
    if any(not env.get(name, "").strip() for name in required):
        raise ValueError("SMTP host, username, and password are required")
    security = env.get("KUANG_SMTP_SECURITY", "ssl").lower()
    if security not in ("ssl", "starttls"):
        raise ValueError("SMTP security must be ssl or starttls")
    port = int(env.get("KUANG_SMTP_PORT", "465" if security == "ssl" else "587"))
    if not 1 <= port <= 65535:
        raise ValueError("Invalid SMTP port")
    sender = env.get("KUANG_SMTP_FROM") or env["KUANG_SMTP_USER"]
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = f"Kuang model completed: {record['run_id']}"
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = make_msgid()
    message.set_content(
        "The requested model workflow completed successfully.\n\n"
        f"Run ID: {record['run_id']}\n"
        f"Output directory: {record['output_directory']}\n"
        f"Started: {record['started_at']}\n"
        f"Completed: {record['completed_at']}\n"
        f"Working directory: {record['working_directory']}\n"
        f"Command: {record['command']}\n"
        f"Run record: {record['record_path']}\n"
    )
    context = ssl.create_default_context()
    if security == "ssl":
        connection = smtplib.SMTP_SSL(env["KUANG_SMTP_HOST"], port,
                                      timeout=20, context=context)
    else:
        connection = smtplib.SMTP(env["KUANG_SMTP_HOST"], port, timeout=20)
    try:
        if security == "starttls":
            connection.ehlo()
            connection.starttls(context=context)
            connection.ehlo()
        connection.login(env["KUANG_SMTP_USER"], env["KUANG_SMTP_PASSWORD"])
        refused = connection.send_message(message, from_addr=sender, to_addrs=[recipient])
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)
    finally:
        # A QUIT failure after DATA was accepted must not turn success into a
        # failure and encourage a duplicate retry. Preserve earlier errors too.
        with suppress(smtplib.SMTPException, OSError):
            connection.quit()
        with suppress(smtplib.SMTPException, OSError):
            connection.close()


def _failure_details(error):
    """Keep numeric SMTP diagnostics and categories, never raw server replies."""
    code = getattr(error, "smtp_code", None)
    reply = getattr(error, "smtp_error", b"")
    if isinstance(error, smtplib.SMTPRecipientsRefused) and error.recipients:
        code, reply = next(iter(error.recipients.values()))
    reply = reply.decode("utf-8", errors="replace") if isinstance(reply, bytes) else str(reply)
    enhanced = re.search(r"(?<!\d)([245]\.\d{1,3}\.\d{1,3})(?!\d)", reply)
    details = {"error_type": type(error).__name__}
    if code is not None:
        details["smtp_code"] = int(code)
    if enhanced:
        details["enhanced_status"] = enhanced.group(1)
    if details.get("enhanced_status") == "5.4.5" or "sending limit" in reply.lower():
        reason = "sending_limit"
    elif any(term in reply.lower() for term in ("rate limit", "too many login attempts", "too many messages")):
        reason = "rate_limited"
    elif code is not None and 400 <= code < 500:
        reason = "temporary_rejection"
    elif isinstance(error, smtplib.SMTPAuthenticationError):
        reason = "authentication_failed"
    elif code is not None and 500 <= code < 600:
        reason = "rejected"
    elif isinstance(error, (OSError, smtplib.SMTPServerDisconnected)):
        reason = "connection_error"
    elif isinstance(error, ValueError):
        reason = "configuration_error"
    else:
        reason = "email_error"
    details["failure_reason"] = reason
    return details


def notify_completed_run(record, *, environ=None):
    """Notify when enabled; mail failures never invalidate a completed simulation.

    Return None when disabled, True for SMTP acceptance, or False on failure.
    SMTP acceptance is not confirmation of inbox delivery.
    """
    env = os.environ if environ is None else environ
    recipient = env.get("KUANG_NOTIFY_EMAIL", "").strip()
    if not recipient:
        return None
    status = {
        "recipient": recipient,
        "attempted_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
    }
    try:
        if record["status"] != "completed":
            raise ValueError("Only completed runs can send completion notifications")
        _send(record, recipient, env)
        status["status"] = "accepted_by_smtp"
        print(f"Email notification accepted by SMTP for {recipient}", flush=True)
        accepted = True
    except Exception as error:
        # Do not log server replies or environment contents: these can contain
        # authentication details. Preserve the numerical run's success status.
        status.update(status="failed", **_failure_details(error))
        print(f"Email notification failed ({type(error).__name__}); "
              f"reason: {status['failure_reason']}. No automatic retry; results are saved. "
              "See EmailNotification.json and the email setup instructions.",
              file=sys.stderr, flush=True)
        accepted = False
    try:
        path = Path(record["record_path"]).with_name("EmailNotification.json")
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    except OSError:
        print("Could not save email notification status.", file=sys.stderr, flush=True)
    return accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send or retry a completed run's email notification.")
    parser.add_argument("record", type=Path, help="Path to an existing RunRecord.json")
    args = parser.parse_args()
    result = notify_completed_run(json.loads(args.record.read_text(encoding="utf-8")))
    if result is None:
        print("Set KUANG_NOTIFY_EMAIL to enable email notifications.", file=sys.stderr)
    sys.exit(0 if result else 1)
