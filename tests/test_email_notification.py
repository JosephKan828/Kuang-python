"""Notification content, encrypted submission, and nonfatal delivery failures."""

import contextlib
import io
import json
from pathlib import Path
import smtplib
import sys
import tempfile
import unittest
from unittest.mock import ANY, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from EmailNotification import notify_completed_run
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


if __name__ == "__main__":
    unittest.main()
