# robot-router: voice/text → Workers AI → the right robot skill

A Cloudflare Worker that turns "please grab the pen" into one of the robot's trained skills. It implements
[docs/architecture.md](../../docs/architecture.md).

```
browser dashboard ──/api/command──► Worker ──► Workers AI: Whisper (speech→text)
                                       │        Mistral Small 3.1 (what's on the table?)
                                       │        Llama 3.3 70B, JSON mode (which skill?)
                                       ▼
                          RobotSession Durable Object ◄──WebSocket── host/robot_agent.py (laptop)
                                                                          │
                                                       robot.py rollout / replay   or   MuJoCo sim (--sim)
```

* The LLM can **only** pick a skill name from the list the agent announced (JSON-schema `enum`), or `none`.
  Confidence < 0.6 → the dashboard asks for confirmation. A busy robot rejects new commands.
* The policy runs on the laptop; Cloudflare gets **one snapshot per command** (scene check) and one after the
  skill (vision "did it work?" check).
* **Live panel** (dashboard WebSocket `/api/live`): every pipeline stage is logged as it happens (heard →
  camera → vision → router → guardrail → run → policy inference → result → vision check). While a sim skill
  runs, the agent also streams ~8 Hz 320×240 frames with the 6 joint positions and the targets the policy
  commands, which the dashboard draws as bars. On the real arm only the log streams for now (lerobot holds the
  camera during a rollout). The log lives in the Durable Object's memory, so it starts empty after it hibernates.
* Every `/api` route needs the shared secret `ROBOT_TOKEN` (`Authorization: Bearer …`, agent and live view: `?token=`).

## Models

| Job | Model | Why |
|---|---|---|
| speech → text | `@cf/openai/whisper-large-v3-turbo` | fast, the dashboard records 16 kHz WAV |
| scene + success check | `@cf/mistralai/mistral-small-3.1-24b-instruct` | vision, **Apache-2.0** |
| instruction → skill | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | supports JSON mode (`response_format`) |

We don't use `llama-3.2-11b-vision-instruct`: accepting its license requires confirming you are *not* an
individual or company based in the EU. That rules it out for this (Austrian) team.

## Skills

Skills come from `config.toml` `[skills.*]` on the laptop. The agent sends `name` + `description` when it
connects, so adding a skill needs no redeploy. Write the description the way you'd explain the skill to a person.
The router reads only that description.

## Run locally

```powershell
cd cloudflare\robot-router
npm install
"ROBOT_TOKEN=dev-token" | Out-File -Encoding ascii .dev.vars
npx wrangler dev                      # http://127.0.0.1:8787 (Workers AI calls are remote, so you need a login)

# second terminal, repo root: the sim stands in for the arm
.venv\Scripts\python host\robot_agent.py --url http://127.0.0.1:8787 --token dev-token --sim --show
```

Open http://127.0.0.1:8787, enter the token, type "please grab the pen" or hold 🎙 and say it.

## Deploy

```powershell
npx wrangler login                    # once (already done on this laptop)
npx wrangler secret put ROBOT_TOKEN   # pick a long random string
npm run deploy                        # -> https://robot-router.<your-subdomain>.workers.dev
```

Then on the robot laptop:
```powershell
.venv\Scripts\python host\robot_agent.py --url https://robot-router.<your-subdomain>.workers.dev --token <ROBOT_TOKEN>
```

## Verified (2026-10-14, `wrangler dev` + sim agent, real Workers AI calls)

| Test | Result |
|---|---|
| no token | 401 |
| agent connects, announces skills | `pick_pen`, `wave`, `reset_scene` |
| "wave to everyone in the audience" | `wave`, confidence 1 → started |
| "put a new pen on the table" | `reset_scene` → started |
| "please grab the pen" | scene "One blue marker, center" → `pick_pen` → started |
| "pick up the red cube" | scene "One blue pen, center" → `none`: "I don't see a red cube to pick up." |
| "make me a coffee" | `none`: "Sorry, I'm not equipped to make coffee." |
| second command while running | rejected: busy |
| after-skill vision check | "No, the pen is still on the table" (correct: the sim policy wasn't trained yet) |

Not yet tested: deployed (`workers.dev`) version, voice input from a real microphone, the real-arm backend.
