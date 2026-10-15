# AustrianStartupsHackathon - HickHack (SO-101 arm)

Three team members, three approaches, one folder each:

| Folder | Who | Approach | Taken from |
|---|---|---|---|
| [`arman/`](arman/) | Arman | Cloud VLA: MolmoAct2 (LeRobot) on a Lightning AI GPU drives the arm from camera images; local code adds a grab check, carry-to-container, retries, wake-up and logs. Also taught poses and joystick tools. | Arman's working copy |
| [`syed/`](syed/) | Syed | His `syed/dev` branch: team task "pick up the pen", calibration, stage runner, self-check, rollout_live. | branch `syed/dev` |
| [`valentino/`](valentino/) | Valentino | Simulation + dashboard + vision check pipeline: instruction to object and box, validated moves (FK + physics replay), Docker/ROS 2 setup. | branch `main` |

Each folder is self-contained; see its own README. The original branches are unchanged.
