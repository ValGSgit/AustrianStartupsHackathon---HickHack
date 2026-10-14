"""Control the robot with the joystick (leader arm). The robot eases to the joystick's pose, then copies it.

    python joystick.py

You can also save poses while driving: type a name + Enter. To stop: q + Enter (or Ctrl+C).
"""

from poses import teach

if __name__ == "__main__":
    teach(use_leader=True)
