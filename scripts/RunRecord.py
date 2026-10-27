"""Record completed launcher invocations and index them beside their outputs."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
from zoneinfo import ZoneInfo


def local_timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="microseconds")


def record_completed_run(output_path, command, started_at, *, log_directory=None):
    """Write one immutable completion record and append it to the parent index.

    Call only after all requested stages succeed. Reusing an output directory
    retains its run ID but creates a separate record for each invocation.
    """
    output_path = Path(output_path).resolve(strict=True)
    if log_directory is None:
        stamp = datetime.fromisoformat(started_at).strftime("%Y-%m-%d-%H-%M-%S-%f")
        log_directory = output_path / "logs" / f"{stamp}-{os.getpid()}"
    log_directory = Path(log_directory).resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    record_path = log_directory / "RunRecord.json"
    history_path = output_path.parent / "RunHistory.jsonl"
    command = [str(arg) for arg in command]
    record = {
        "status": "completed",
        "run_id": output_path.name,
        "invocation_id": log_directory.name,
        "output_directory": str(output_path),
        "started_at": started_at,
        "completed_at": local_timestamp(),
        "timezone": "Asia/Taipei",
        "working_directory": str(Path.cwd()),
        "command": shlex.join(command),
        "argv": command,
        "record_path": str(record_path),
    }
    with record_path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)
        file.write("\n")
    # Parallel runs may complete together; keep each JSON line intact.
    with history_path.open("a", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        fcntl.flock(file, fcntl.LOCK_UN)
    print(f"Run record: {record_path}", flush=True)
    print(f"Run history: {history_path}", flush=True)
    return record_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--log-directory", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply the launch command after --.")
    record_completed_run(args.output_dir, command, args.started_at,
                         log_directory=args.log_directory)
