"""Notification content, encrypted submission, and nonfatal delivery failures."""

import contextlib
import io
import getpass
import json
import os
from pathlib import Path
import smtplib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import ANY, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from EmailNotification import main as notification_main, notify_completed_run
from RunRecord import record_completed_run


class EmailNotificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.record = {
            "status": "completed", "run_id": "test-run", "output_directory": str(self.output),
            "started_at": "2026-10-27T12:00:00+08:00",
            "completed_at": "2026-10-27T13:00:00+08:00",
            "working_directory": "/work/model", "command": "bash scripts/RunScript.sh --seed 42",
            "record_path": str(self.output / "RunRecord.json"),
        }
        self.env = {
            "KUANG_NOTIFY_EMAIL": "recipient@example.com", "KUANG_SMTP_HOST": "smtp.example.com",
            "KUANG_SMTP_USER": "sender@example.com", "KUANG_SMTP_PASSWORD": "test-secret",
        }

    def test_disabled_notification_never_connects(self):
        with patch("EmailNotification.smtplib.SMTP_SSL") as smtp_ssl, \
             patch("EmailNotification.smtplib.SMTP") as smtp:
            self.assertIsNone(notify_completed_run(self.record, environ={}))
        smtp_ssl.assert_not_called()
        smtp.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_prompt_supplies_password_to_the_sending_process(self):
        path = self.output / "RunRecord.json"
        path.write_text(json.dumps(self.record))
        env = dict(self.env, KUANG_SMTP_PASSWORD="")
        stdout = io.StringIO()
        with patch.dict("os.environ", env, clear=True), \
             patch("EmailNotification.getpass.getpass", return_value="typed-secret") as prompt, \
             patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
             contextlib.redirect_stdout(stdout):
            smtp.return_value.send_message.return_value = {}
            self.assertEqual(notification_main(["--prompt-password", str(path)]), 0)
            prompt.assert_called_once()
            smtp.return_value.login.assert_called_once_with("sender@example.com", "typed-secret")
        self.assertNotIn("typed-secret", stdout.getvalue() + (self.output / "EmailNotification.json").read_text())

    def test_prompt_refuses_visible_input_without_a_terminal(self):
        with patch("EmailNotification.getpass.getpass", side_effect=getpass.GetPassWarning), \
             patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(notification_main(["--prompt-password", "--check-config"]), 1)
            smtp.assert_not_called()

    def test_ssl_message_contains_run_details_but_no_credentials(self):
        with patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
             contextlib.redirect_stdout(io.StringIO()):
            server = smtp.return_value
            server.send_message.return_value = {}
            self.assertTrue(notify_completed_run(self.record, environ=self.env))
        smtp.assert_called_once_with("smtp.example.com", 465, timeout=20, context=ANY)
        self.assertTrue(smtp.call_args.kwargs["context"].check_hostname)
        server.login.assert_called_once_with("sender@example.com", "test-secret")
        message = server.send_message.call_args.args[0]
        self.assertEqual(message["To"], "recipient@example.com")
        self.assertEqual(message["From"], "sender@example.com")
        for key in ("run_id", "output_directory", "started_at", "completed_at", "command"):
            self.assertIn(self.record[key], message.get_content())
        status = (self.output / "EmailNotification.json").read_text()
        self.assertEqual(json.loads(status)["status"], "accepted_by_smtp")
        self.assertNotIn("test-secret", message.as_string() + status)

    def test_starttls_precedes_authentication(self):
        self.env["KUANG_SMTP_SECURITY"] = "starttls"
        with patch("EmailNotification.smtplib.SMTP") as smtp, \
             contextlib.redirect_stdout(io.StringIO()):
            server = smtp.return_value
            server.send_message.return_value = {}
            self.assertTrue(notify_completed_run(self.record, environ=self.env))
        smtp.assert_called_once_with("smtp.example.com", 587, timeout=20)
        server.assert_has_calls([
            call.ehlo(), call.starttls(context=ANY), call.ehlo(),
            call.login("sender@example.com", "test-secret"),
            call.send_message(ANY, from_addr="sender@example.com", to_addrs=["recipient@example.com"]),
        ])

    def test_invalid_configuration_never_connects(self):
        for updates in ({"KUANG_SMTP_PASSWORD": ""}, {"KUANG_SMTP_SECURITY": "plain"},
                        {"KUANG_SMTP_PORT": "invalid"}, {"KUANG_SMTP_PORT": "0"}):
            with self.subTest(updates=updates), \
                 patch("EmailNotification.smtplib.SMTP_SSL") as smtp_ssl, \
                 patch("EmailNotification.smtplib.SMTP") as smtp, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(notify_completed_run(self.record, environ=dict(self.env, **updates)))
                smtp_ssl.assert_not_called()
                smtp.assert_not_called()

    def test_configuration_errors_identify_settings_without_exposing_values(self):
        for updates, expected in (
            ({"KUANG_SMTP_PASSWORD": ""}, "KUANG_SMTP_PASSWORD"),
            ({"KUANG_SMTP_HOST": "", "KUANG_SMTP_USER": ""}, "KUANG_SMTP_HOST, KUANG_SMTP_USER"),
            ({"KUANG_SMTP_PORT": "test-secret"}, "KUANG_SMTP_PORT"),
            ({"KUANG_SMTP_SECURITY": "test-secret"}, "KUANG_SMTP_SECURITY"),
            ({"KUANG_SMTP_USER": "your-sending-account@gmail.com"}, "Replace the example KUANG_SMTP_USER"),
        ):
            stderr = io.StringIO()
            with self.subTest(expected=expected), patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
                 contextlib.redirect_stderr(stderr):
                self.assertFalse(notify_completed_run(self.record, environ=dict(self.env, **updates)))
                smtp.assert_not_called()
                text = (self.output / "EmailNotification.json").read_text()
                self.assertIn(expected, json.loads(text)["configuration_error"])
                self.assertIn(expected, stderr.getvalue())
                self.assertNotIn("test-secret", text + stderr.getvalue())

    def test_configuration_check_cli_needs_no_run_or_connection(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "EmailNotification.py"
        for password, expected_code in (("test-secret", 0), ("", 1)):
            with self.subTest(configured=bool(password)):
                env = dict(os.environ, **self.env)
                env["KUANG_SMTP_PASSWORD"] = password
                result = subprocess.run([sys.executable, str(script), "--check-config"],
                                        env=env, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, expected_code, result.stderr)
                if expected_code:
                    self.assertIn("KUANG_SMTP_PASSWORD", result.stderr)
                else:
                    self.assertIn("authentication and delivery are not tested", result.stdout)
                self.assertNotIn("test-secret", result.stdout + result.stderr)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_mail_failure_preserves_completed_run_and_redacts_server_reply(self):
        run = self.output / "run"
        run.mkdir()
        logs = run / "logs" / "invocation"
        stderr = io.StringIO()
        with patch.dict("os.environ", self.env, clear=True), \
             patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            def reject(*args, **kwargs):
                self.assertTrue((logs / "RunRecord.json").exists())
                self.assertTrue((self.output / "RunHistory.jsonl").exists())
                raise smtplib.SMTPAuthenticationError(535, b"test-secret")
            smtp.return_value.login.side_effect = reject
            result = record_completed_run(run, ["bash", "RunScript.sh"],
                                          self.record["started_at"], log_directory=logs)
            self.assertFalse(notify_completed_run(json.loads(result.read_text())))
        self.assertEqual(json.loads(result.read_text())["status"], "completed")
        notification = (logs / "EmailNotification.json").read_text()
        self.assertEqual(json.loads(notification)["status"], "failed")
        self.assertIn("SMTPAuthenticationError", stderr.getvalue())
        self.assertNotIn("test-secret", stderr.getvalue() + notification + result.read_text())

    def test_recipient_refusal_is_not_reported_as_acceptance(self):
        with patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
             contextlib.redirect_stderr(io.StringIO()):
            smtp.return_value.send_message.return_value = {
                "recipient@example.com": (550, b"rejected"),
            }
            self.assertFalse(notify_completed_run(self.record, environ=self.env))
        self.assertEqual(json.loads((self.output / "EmailNotification.json").read_text())["status"], "failed")

    def test_sending_limit_preserves_completed_run_without_retry(self):
        for name, method, error, reason in (
            ("daily", "send_message", smtplib.SMTPDataError(550, b"5.4.5 Daily user sending limit exceeded test-secret"), "sending_limit"),
            ("login", "login", smtplib.SMTPAuthenticationError(454, b"4.7.0 Too many login attempts test-secret"), "rate_limited"),
            ("temporary", "send_message", smtplib.SMTPDataError(451, b"4.3.0 Temporary failure test-secret"), "temporary_rejection"),
        ):
            run = self.output / name
            run.mkdir()
            stderr = io.StringIO()
            with self.subTest(reason=reason), patch.dict("os.environ", self.env, clear=True), \
                 patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                getattr(smtp.return_value, method).side_effect = error
                path = record_completed_run(run, ["bash", "RunScript.sh"], self.record["started_at"])
                self.assertFalse(notify_completed_run(json.loads(path.read_text())))
                smtp.assert_called_once()
                smtp.return_value.login.assert_called_once()
                self.assertEqual(smtp.return_value.send_message.call_count, int(method == "send_message"))
                self.assertEqual(json.loads(path.read_text())["status"], "completed")
                status_text = path.with_name("EmailNotification.json").read_text()
                status = json.loads(status_text)
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["smtp_code"], error.smtp_code)
                self.assertEqual(status["failure_reason"], reason)
                self.assertNotIn("test-secret", status_text + stderr.getvalue())

    def test_logout_failure_does_not_undo_smtp_acceptance(self):
        with patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
             contextlib.redirect_stdout(io.StringIO()):
            smtp.return_value.send_message.return_value = {}
            smtp.return_value.quit.side_effect = smtplib.SMTPResponseException(421, b"closing connection")
            self.assertTrue(notify_completed_run(self.record, environ=self.env))
            smtp.return_value.send_message.assert_called_once()
            smtp.return_value.close.assert_called_once()
        self.assertEqual(json.loads((self.output / "EmailNotification.json").read_text())["status"], "accepted_by_smtp")

    def test_connection_timeout_is_nonfatal(self):
        with patch("EmailNotification.smtplib.SMTP_SSL", side_effect=TimeoutError) as smtp, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(notify_completed_run(self.record, environ=self.env))
            smtp.assert_called_once()
        self.assertEqual(json.loads((self.output / "EmailNotification.json").read_text())["failure_reason"], "connection_error")

    def test_connection_check_does_not_need_credentials_or_send(self):
        for security, smtp_name in (("ssl", "SMTP_SSL"), ("starttls", "SMTP")):
            env = {"KUANG_SMTP_HOST": "smtp.example.com", "KUANG_SMTP_SECURITY": security}
            with self.subTest(security=security), patch.dict("os.environ", env, clear=True), \
                 patch(f"EmailNotification.smtplib.{smtp_name}") as smtp, \
                 patch("EmailNotification.getpass.getpass") as prompt, \
                 contextlib.redirect_stdout(io.StringIO()):
                smtp.return_value.ehlo.return_value = (250, b"OK")
                self.assertEqual(notification_main(["--check-connection"]), 0)
                smtp.return_value.login.assert_not_called()
                smtp.return_value.send_message.assert_not_called()
                prompt.assert_not_called()
                smtp.return_value.close.assert_called_once()

    def test_disconnect_reports_the_stage_without_exposing_server_text(self):
        for stage in ("connect", "authenticate", "submit"):
            stderr = io.StringIO()
            with self.subTest(stage=stage), patch("EmailNotification.smtplib.SMTP_SSL") as smtp, \
                 contextlib.redirect_stderr(stderr):
                error = smtplib.SMTPServerDisconnected("test-secret")
                if stage == "connect":
                    smtp.side_effect = error
                else:
                    method = "login" if stage == "authenticate" else "send_message"
                    getattr(smtp.return_value, method).side_effect = error
                self.assertFalse(notify_completed_run(self.record, environ=self.env))
                text = (self.output / "EmailNotification.json").read_text()
                status = json.loads(text)
                self.assertEqual(status["smtp_stage"], stage)
                self.assertEqual(status.get("delivery_uncertain", False), stage == "submit")
                self.assertIn(f"SMTP stage: {stage}", stderr.getvalue())
                self.assertNotIn("test-secret", stderr.getvalue() + text)

    def test_starttls_disconnect_closes_connection_and_reports_stage(self):
        env = dict(self.env, KUANG_SMTP_SECURITY="starttls")
        with patch("EmailNotification.smtplib.SMTP") as smtp, \
             contextlib.redirect_stderr(io.StringIO()):
            smtp.return_value.starttls.side_effect = smtplib.SMTPServerDisconnected()
            self.assertFalse(notify_completed_run(self.record, environ=env))
            smtp.return_value.close.assert_called_once()
            smtp.return_value.login.assert_not_called()
        self.assertEqual(json.loads((self.output / "EmailNotification.json").read_text())["smtp_stage"], "starttls")


if __name__ == "__main__":
    unittest.main()
