"""Optional SMTP notifications for completed runs; credentials come from the environment."""

import argparse
from contextlib import suppress
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import getpass
import json
import os
from pathlib import Path
import re
import smtplib
import ssl
import sys
import warnings
from zoneinfo import ZoneInfo


class EmailConfigurationError(ValueError):
    """A configuration error with a message safe to show without redaction."""


def check_configuration(env, *, require_credentials=True):
    """Validate exported settings without connecting or exposing their values."""
    required = ("KUANG_SMTP_HOST", "KUANG_SMTP_USER", "KUANG_SMTP_PASSWORD") if require_credentials else ("KUANG_SMTP_HOST",)
    missing = [name for name in required if not env.get(name, "").strip()]
    if missing:
        raise EmailConfigurationError("Missing or empty exported settings: " + ", ".join(missing))
    if require_credentials and env["KUANG_SMTP_USER"].strip().lower() in (
        "your-sending-account@gmail.com", "your_sending_account@gmail.com",
    ):
        raise EmailConfigurationError("Replace the example KUANG_SMTP_USER with the sending account")
    security = env.get("KUANG_SMTP_SECURITY", "ssl").strip().lower()
    if security not in ("ssl", "starttls"):
        raise EmailConfigurationError("KUANG_SMTP_SECURITY must be ssl or starttls")
    try:
        port = int(env.get("KUANG_SMTP_PORT", "465" if security == "ssl" else "587"))
    except ValueError:
        raise EmailConfigurationError("KUANG_SMTP_PORT must be an integer from 1 to 65535") from None
    if not 1 <= port <= 65535:
        raise EmailConfigurationError("KUANG_SMTP_PORT must be an integer from 1 to 65535")
    return security, port


def _close_connection(connection):
    # Errors during logout must not undo acceptance or mask an earlier failure.
    with suppress(smtplib.SMTPException, OSError):
        connection.quit()
    with suppress(smtplib.SMTPException, OSError):
        connection.close()


def _open_connection(env, security, port):
    context = ssl.create_default_context()
    connection = None
    stage = "connect"
    try:
        if security == "ssl":
            connection = smtplib.SMTP_SSL(env["KUANG_SMTP_HOST"], port,
                                          timeout=20, context=context)
        else:
            connection = smtplib.SMTP(env["KUANG_SMTP_HOST"], port, timeout=20)
            stage = "ehlo"
            connection.ehlo()
            stage = "starttls"
            connection.starttls(context=context)
            stage = "ehlo"
            connection.ehlo()
    except Exception as error:
        error.smtp_stage = stage
        if connection is not None:
            with suppress(smtplib.SMTPException, OSError):
                connection.close()
        raise
    return connection


def check_connection(env):
    """Verify SMTP/TLS reachability without a password, login, or email."""
    security, port = check_configuration(env, require_credentials=False)
    connection = _open_connection(env, security, port)
    try:
        code, reply = connection.ehlo()
        if code != 250:
            raise smtplib.SMTPHeloError(code, reply)
    except Exception as error:
        error.smtp_stage = "ehlo"
        raise
    finally:
        _close_connection(connection)


def _send(record, recipient, env):
    security, port = check_configuration(env)
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
    connection = _open_connection(env, security, port)
    stage = "authenticate"
    try:
        connection.login(env["KUANG_SMTP_USER"], env["KUANG_SMTP_PASSWORD"])
        stage = "submit"
        refused = connection.send_message(message, from_addr=sender, to_addrs=[recipient])
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)
    except Exception as error:
        error.smtp_stage = stage
        raise
    finally:
        # A QUIT failure after DATA was accepted must not turn success into a
        # failure and encourage a duplicate retry. Preserve earlier errors too.
        _close_connection(connection)


def _failure_details(error):
    """Keep numeric SMTP diagnostics and categories, never raw server replies."""
    code = getattr(error, "smtp_code", None)
    reply = getattr(error, "smtp_error", b"")
    if isinstance(error, smtplib.SMTPRecipientsRefused) and error.recipients:
        code, reply = next(iter(error.recipients.values()))
    reply = reply.decode("utf-8", errors="replace") if isinstance(reply, bytes) else str(reply)
    enhanced = re.search(r"(?<!\d)([245]\.\d{1,3}\.\d{1,3})(?!\d)", reply)
    details = {"error_type": type(error).__name__}
    if hasattr(error, "smtp_stage"):
        details["smtp_stage"] = error.smtp_stage
    if isinstance(error, EmailConfigurationError):
        details["configuration_error"] = str(error)
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
        if "configuration_error" in status:
            print(status["configuration_error"], file=sys.stderr, flush=True)
        if "smtp_stage" in status:
            print(f"SMTP stage: {status['smtp_stage']}", file=sys.stderr, flush=True)
            if status["smtp_stage"] == "submit" and status["failure_reason"] == "connection_error":
                status["delivery_uncertain"] = True
                print("Delivery is uncertain; check the inbox before retrying.", file=sys.stderr, flush=True)
        accepted = False
    try:
        path = Path(record["record_path"]).with_name("EmailNotification.json")
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    except OSError:
        print("Could not save email notification status.", file=sys.stderr, flush=True)
    return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description="Send or retry a completed run's email notification.")
    parser.add_argument("record", type=Path, nargs="?", help="Path to an existing RunRecord.json")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate exported email settings without connecting or sending")
    parser.add_argument("--prompt-password", action="store_true",
                        help="Read the SMTP app password privately for this invocation")
    parser.add_argument("--check-connection", action="store_true",
                        help="Check SMTP/TLS reachability without logging in or sending")
    args = parser.parse_args(argv)
    if args.check_connection:
        if args.prompt_password:
            parser.error("--check-connection does not need --prompt-password.")
        try:
            check_connection(os.environ)
        except Exception as error:
            details = _failure_details(error)
            print(f"SMTP connection check failed ({details['error_type']}); "
                  f"stage: {details.get('smtp_stage', 'configuration')}; "
                  f"reason: {details['failure_reason']}.", file=sys.stderr)
            if "configuration_error" in details:
                print(details["configuration_error"], file=sys.stderr)
            return 1
        print("SMTP connection and TLS verification succeeded. No login or email was attempted.")
        return 0
    if args.record is None and not args.check_config:
        parser.error("Supply a RunRecord.json path, or use --check-config or --check-connection.")
    if args.prompt_password:
        try:
            # Refuse getpass's fallback to visible input when no terminal is
            # available. The password exists only in this process environment.
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                os.environ["KUANG_SMTP_PASSWORD"] = getpass.getpass("SMTP app password: ")
        except (EOFError, getpass.GetPassWarning):
            print("Could not read a hidden password; run this command in an interactive terminal.",
                  file=sys.stderr)
            return 1
    if args.check_config:
        try:
            if not os.environ.get("KUANG_NOTIFY_EMAIL", "").strip():
                raise EmailConfigurationError("Missing or empty exported setting: KUANG_NOTIFY_EMAIL")
            check_configuration(os.environ)
        except EmailConfigurationError as error:
            print(f"Email configuration error: {error}", file=sys.stderr)
            return 1
        print("Email configuration is present and valid; authentication and delivery are not tested.")
        return 0
    result = notify_completed_run(json.loads(args.record.read_text(encoding="utf-8")))
    if result is None:
        print("Set KUANG_NOTIFY_EMAIL to enable email notifications.", file=sys.stderr)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
