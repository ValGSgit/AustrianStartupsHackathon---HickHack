"""`lerobot-train` that can save checkpoints on Windows without admin rights.

LeRobot points `checkpoints/last` at the newest checkpoint with a symlink. Windows only
allows symlinks with admin rights or Developer Mode, so every checkpoint save crashes with
WinError 1314. This wrapper swaps in a directory junction, which any user can create and
which `--resume` and `robot.py rollout` follow like a symlink. All arguments are passed to
lerobot-train unchanged.
"""

import os
import sys
from pathlib import Path

from lerobot.common import train_utils
from lerobot.scripts import lerobot_train


def update_last_checkpoint(checkpoint_dir: Path) -> None:
    last = checkpoint_dir.parent / train_utils.LAST_CHECKPOINT_LINK
    if last.is_junction() or last.is_symlink():
        os.unlink(last) if last.is_symlink() else os.rmdir(last)  # removes the link, not the target
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(checkpoint_dir.resolve()), str(last))
    else:
        last.symlink_to(checkpoint_dir.relative_to(checkpoint_dir.parent))


train_utils.update_last_checkpoint = update_last_checkpoint
lerobot_train.update_last_checkpoint = update_last_checkpoint

if __name__ == "__main__":
    lerobot_train.main()
