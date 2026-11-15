"""Read-only AutoDL diagnostics over OpenSSH; passwords are entered only in SSH."""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import time
from pathlib import Path


REMOTE_CODE = r'''
import importlib.metadata, json, os, pathlib, platform, shutil, subprocess, sys
root = pathlib.Path(__ROOT__)
def run(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"error": str(exc)}
def versions(names):
    result = {}
    for name in names:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    return result
inventory = {}
for folder in ("dataset/archives", "dataset/processed", "dataset/weights", "outputs", "ablation"):
    directory = root / folder
    files = []
    if directory.exists():
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (folder != "ablation" or path.suffix in (".json", ".csv")):
                files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
                if len(files) >= 200: break
    inventory[folder] = files
disk = shutil.disk_usage(root if root.exists() else pathlib.Path.home())
payload = {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version,
           "python_executable": sys.executable, "root": str(root),
           "packages": versions(["torch", "torchvision", "open-clip-torch", "transformers", "accelerate", "PyYAML", "numpy"]),
           "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
           "disk_bytes": {"total": disk.total, "used": disk.used, "free": disk.free},
           "git_head": run(["git", "-C", str(root), "rev-parse", "HEAD"]),
           "git_status": run(["git", "-C", str(root), "status", "--short", "--branch"]),
           "git_remote": run(["git", "-C", str(root), "remote", "-v"]),
           "inventory": inventory}
print(json.dumps(payload, ensure_ascii=False))
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="connect.westb.seetacloud.com")
    parser.add_argument("--port", type=int, default=32132)
    parser.add_argument("--user", default="root")
    parser.add_argument("--root", default="/root/autodl-tmp")
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--identity-file", help="Optional existing SSH key; never a password")
    parser.add_argument("--interactive-password", action="store_true", help="Allow SSH itself to request a password in the terminal")
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--retries", type=int, default=1, help="Additional attempts after initial failure")
    parser.add_argument("--output", help="Optional local diagnostic JSON file")
    args = parser.parse_args(argv)
    if args.retries < 0 or args.retries > 3 or args.connect_timeout <= 0:
        parser.error("Use retries 0..3 and a positive connect timeout")
    code = REMOTE_CODE.replace("__ROOT__", repr(args.root))
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    snippet = f"import base64; exec(base64.b64decode({encoded!r}))"
    remote = shlex.quote(args.python) + " -c " + shlex.quote(snippet)
    command = ["ssh", "-p", str(args.port), "-o", f"ConnectTimeout={args.connect_timeout}",
               "-o", "StrictHostKeyChecking=accept-new"]
    if not args.interactive_password:
        command += ["-o", "BatchMode=yes"]
    if args.identity_file:
        command += ["-i", args.identity_file]
    command += [f"{args.user}@{args.host}", remote]
    result = None
    for attempt in range(args.retries + 1):
        # In interactive mode stdin/stderr stay attached so OpenSSH prompts directly.
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=None if args.interactive_password else subprocess.PIPE)
        if result.returncode == 0:
            break
        if attempt < args.retries:
            print(f"SSH check failed; retrying once ({attempt+1}/{args.retries})", flush=True)
            time.sleep(1)
    if result is None or result.returncode:
        message = result.stderr.strip() if result and result.stderr else "SSH failed; check endpoint, instance state and authentication"
        parser.exit(result.returncode if result else 1, message + "\n")
    try:
        report = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        parser.exit(1, "Remote output was not diagnostic JSON; inspect login-shell output\n")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
