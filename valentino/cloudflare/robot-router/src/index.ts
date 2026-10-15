// robot-router Worker: voice/text instruction -> Workers AI -> one of the robot's trained skills.
//
//   GET  /api/agent          WebSocket for the laptop agent (host/robot_agent.py)
//   GET  /api/live           WebSocket for dashboards: live camera frames, joint targets, pipeline log
//   GET  /api/state          robot + last command / result
//   GET  /api/image/:kind    before | after snapshot (JPEG)
//   POST /api/command        {text} or {audio: base64 wav}  -> transcribe, look, route, guard, run
//   POST /api/run            {skill}   run a skill directly (confirm button / bypass when AI is down)
//   POST /api/stop           stop the running skill
//   POST /api/hw             {action}  camera_on | camera_off | arm_on | arm_off  (laptop devices, status line)
//   POST /api/look           one camera frame -> vision model, shown in the Live panel (no skill runs)
//   POST /api/intent         {text}  camera frame + "move X to box B" -> object, box, agreement, GO/ASK (no skill runs)
// All /api routes need the shared secret: "Authorization: Bearer <ROBOT_TOKEN>" (agent: ?token=...).

import { describeScene, inferIntent, route, transcribe, type Intent } from "./ai";
import { RobotSession, type CommandLog, type Env } from "./session";

export { RobotSession };

const CONFIDENCE_MIN = 0.6;
const HW_ACTIONS = ["camera_on", "camera_off", "arm_on", "arm_off"];

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });

function authorized(req: Request, env: Env): boolean {
  if (!env.ROBOT_TOKEN) return false; // refuse everything until the secret is set
  const url = new URL(req.url);
  const header = req.headers.get("Authorization") ?? "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : url.searchParams.get("token");
  return token === env.ROBOT_TOKEN;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (!url.pathname.startsWith("/api/")) return env.ASSETS.fetch(req);
    if (!authorized(req, env)) return json({ error: "unauthorized" }, 401);

    const robot = env.ROBOT.get(env.ROBOT.idFromName("so101"));

    if (url.pathname === "/api/agent" || url.pathname === "/api/live") return robot.fetch(req);
    if (url.pathname === "/api/state" && req.method === "GET") return json(await robot.getState());

    const img = url.pathname.match(/^\/api\/image\/(before|after)$/);
    if (img && req.method === "GET") {
      const b64 = await robot.getImage(img[1] as "before" | "after");
      if (!b64) return new Response("no image", { status: 404 });
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      return new Response(bytes, { headers: { "content-type": "image/jpeg", "cache-control": "no-store" } });
    }

    if (req.method !== "POST") return json({ error: "not found" }, 404);
    const body = (await req.json().catch(() => ({}))) as { text?: string; audio?: string; skill?: string; task?: string };

    if (url.pathname === "/api/stop") return json({ sent: await robot.stop() });

    if (url.pathname === "/api/hw") {
      const action = String((body as { action?: string }).action ?? "");
      if (!HW_ACTIONS.includes(action)) return json({ error: `action must be one of ${HW_ACTIONS.join(", ")}` }, 400);
      return json({ sent: await robot.hw(action) });
    }

    if (url.pathname === "/api/look") return json(await look(env, robot));

    if (url.pathname === "/api/intent") return json(await intent(env, robot, String(body.text ?? "").trim()));

    if (url.pathname === "/api/run") {
      if (!body.skill) return json({ error: "skill required" }, 400);
      const r = await robot.run(body.skill, body.task ?? "");
      // a button press is a command too, so "Last command" and the result always describe the same run
      await robot.logCommand({
        at: new Date().toISOString(), source: "button", transcript: "", scene: "",
        decision: { policy: body.skill, confidence: 1, say: `Running ${body.skill}.`, task: body.task ?? "" },
        action: r.ok ? "started" : r.reason.startsWith("busy") ? "busy" : "error",
        ...(r.ok ? {} : { error: r.reason }),
      });
      return json(r);
    }

    if (url.pathname === "/api/command") return json(await handleCommand(env, robot, body));

    return json({ error: "not found" }, 404);
  },
} satisfies ExportedHandler<Env>;

/** Vision showcase: what does the model see right now? Same snapshot + prompt as a real command, no skill runs. */
async function look(env: Env, robot: DurableObjectStub<RobotSession>): Promise<{ scene: string; ms: number; error?: string }> {
  const state = await robot.getState();
  if (!state.agentConnected) return { scene: "", ms: 0, error: "robot agent is not connected" };
  if (!(await robot.allowAi())) {
    await robot.log("error", "too many AI requests this minute - wait a moment");
    return { scene: "", ms: 0, error: "rate limited, try again in a minute" };
  }
  let t = Date.now();
  const jpeg = await robot.snapshot(5000);
  await robot.log("camera", jpeg ? `look: snapshot from the robot camera (${Date.now() - t} ms)` : "look: no usable camera frame");
  if (!jpeg) return { scene: "", ms: 0, error: "no camera frame" };
  t = Date.now();
  try {
    const scene = await describeScene(env.AI, jpeg, state.mode);
    const ms = Date.now() - t;
    await robot.log("vision", `Mistral Small 3.1 sees: ${scene} (${ms} ms)`);
    return { scene, ms };
  } catch (e) {
    await robot.log("error", `vision failed: ${e}`);
    return { scene: "", ms: 0, error: String(e) };
  }
}

/** Intent check: does the vision model understand which object and which drawn box? Nothing moves. */
async function intent(
  env: Env, robot: DurableObjectStub<RobotSession>, text: string,
): Promise<(Intent & { jpeg: string; instruction: string }) | { error: string }> {
  if (!text) return { error: "type an instruction, e.g. 'move the orange disc to box B'" };
  const state = await robot.getState();
  if (!state.agentConnected) return { error: "robot agent is not connected (it provides the camera frame)" };
  if (!(await robot.allowAi())) return { error: "rate limited, try again in a minute" };
  let t = Date.now();
  const jpeg = await robot.snapshot(5000);
  if (!jpeg) {
    await robot.log("error", "intent: no usable camera frame");
    return { error: "no camera frame (camera index? lens covered?)" };
  }
  await robot.log("camera", `intent: snapshot from the robot camera (${Date.now() - t} ms)`);
  await robot.log("intent", `"${text}" -> Mistral Small 3.1, 5 answers in parallel…`);
  t = Date.now();
  const r = await inferIntent(env.AI, jpeg, text);
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  await robot.log("intent", `${r.verdict}: ${r.answer.object ?? "?"} -> box ${r.box} · agreement ${pct(r.agreement)} ` +
    `(box ${pct(r.boxAgreement)}, object ${pct(r.objectAgreement)})` +
    (r.selfConfidence !== null ? ` · self ${Math.round(r.selfConfidence)}%` : "") + ` (${Date.now() - t} ms)`);
  if (r.reasons.length) await robot.log("intent", `would ask a human: ${r.reasons.join("; ")}`);
  return { ...r, jpeg, instruction: text };
}

async function handleCommand(
  env: Env,
  robot: DurableObjectStub<RobotSession>,
  body: { text?: string; audio?: string },
): Promise<CommandLog & { snapshot: boolean }> {
  const log: CommandLog = {
    at: new Date().toISOString(), source: body.audio ? "voice" : "text", transcript: "", scene: "", decision: null, action: "error",
  };
  let snapshot = false;
  const since = (t: number) => `${Date.now() - t} ms`;
  try {
    // 1. what was said
    let t = Date.now();
    if (body.audio) {
      await robot.log("heard", "audio received, transcribing with Whisper large-v3-turbo…");
      log.transcript = await transcribe(env.AI, body.audio);
      await robot.log("heard", `"${log.transcript}" (${since(t)})`);
    } else {
      log.transcript = (body.text ?? "").trim();
      await robot.log("heard", `typed: "${log.transcript}"`);
    }
    if (!log.transcript) throw new Error("empty instruction");

    const state = await robot.getState();
    if (!state.agentConnected) {
      log.action = "no_agent";
      await robot.log("error", "no robot agent connected - nothing to run");
      await robot.logCommand(log);
      return { ...log, snapshot };
    }
    if (!(await robot.allowAi())) throw new Error("too many AI requests this minute - wait a moment");

    // 2. what the camera sees (one frame, same camera the policy uses)
    t = Date.now();
    const jpeg = await robot.snapshot(5000);
    snapshot = !!jpeg;
    await robot.log("camera", jpeg ? `snapshot from the robot camera (${since(t)})` : "no usable camera frame - routing without a scene");
    if (jpeg) {
      t = Date.now();
      log.scene = await describeScene(env.AI, jpeg, state.mode).catch((e) => `(vision failed: ${e})`);
      await robot.log("vision", `Mistral Small 3.1 sees: ${log.scene} (${since(t)})`);
    }

    // 3. which skill - the LLM can only pick from the agent's list
    t = Date.now();
    await robot.log("router", `Llama 3.3 70B picks one of [${state.skills.map((s) => s.name).join(", ")}, none]…`);
    log.decision = await route(env.AI, log.transcript, state.skills, log.scene);
    const pct = (log.decision.confidence * 100).toFixed(0);
    await robot.log("router", `-> ${log.decision.policy} (confidence ${pct}%) "${log.decision.say}" (${since(t)})`);
    if (log.decision.task) await robot.log("router", `instruction for the policy: "${log.decision.task}"`);

    // 4. guardrails, in code rather than in the prompt
    if (log.decision.policy === "none") log.action = "none";
    else if (log.decision.confidence < CONFIDENCE_MIN) log.action = "needs_confirm";
    else {
      const r = await robot.run(log.decision.policy, log.decision.task);
      log.action = r.ok ? "started" : r.reason.startsWith("busy") ? "busy" : "error";
      if (!r.ok) log.error = r.reason;
    }
    const guard: Record<string, string> = {
      none: "no matching skill - nothing runs",
      needs_confirm: `confidence below ${CONFIDENCE_MIN * 100}% - waiting for a human to confirm`,
      started: `confidence >= ${CONFIDENCE_MIN * 100}% - running it`,
      busy: `robot is busy - ${log.error}`,
      error: `could not start - ${log.error}`,
    };
    await robot.log("guard", guard[log.action] ?? log.action);
  } catch (e) {
    log.action = "error";
    log.error = String(e);
    await robot.log("error", log.error);
  }
  await robot.logCommand(log);
  return { ...log, snapshot };
}
