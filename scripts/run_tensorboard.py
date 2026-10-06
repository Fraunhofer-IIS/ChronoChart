import subprocess
import sys

from src.localization.config.paths import PathConfig

pc = PathConfig()

logdir = pc.lightning_logs  # or "runs" or your custom log path
port = 10005
print(logdir)
subprocess.run(
    [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--port", str(port)],
    check=True,
)


#ps aux | grep tensorboard
