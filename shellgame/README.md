# Shell game: the robot magician

The robot shows the ball, shuffles 3 cups between 4 taped slots by sliding them, the judge guesses,
the robot lifts the right cup and the camera confirms the ball. Interfaces: [CONTRACT.md](CONTRACT.md).

Props: 3 small light identical cups (one bright colour), a soft ball in a contrasting colour (e.g. blue cups,
orange pom-pom). Slots on an arc around the base, same distance, more than 2.6 cup radii apart.

```powershell
# no hardware needed
.venv\Scripts\python shellgame\test_game.py
.venv\Scripts\python shellgame\test_vision.py
.venv\Scripts\python shellgame\shuffle.py --fake --moves 7
.venv\Scripts\python shellgame\show.py --fake --no-vision

# with the arms (follower + leader) and cups on the table
.venv\Scripts\python shellgame\teach.py teach          # guided: home, ready, above/grip/in/lift per slot
.venv\Scripts\python shellgame\teach.py list           # what is still missing
.venv\Scripts\python shellgame\teach.py test-slot 1    # slow step-by-step check of one slot
.venv\Scripts\python shellgame\shuffle.py --slow-test  # first real shuffle, Enter between moves

# camera (fixed above the slots)
.venv\Scripts\python shellgame\vision.py setup         # click slot centres, cup edge, a cup, the ball; s = save
.venv\Scripts\python shellgame\vision.py live

# the show
.venv\Scripts\python shellgame\show.py --fullscreen    # SPACE start, 1-4 guess, ESC stop, q quit
```

Every script ends by moving slowly to `home`, then switching the motors off. Re-teach one pose with
`teach.py teach --only grip_2`. Games are logged to `evals/shell_log.csv`.
