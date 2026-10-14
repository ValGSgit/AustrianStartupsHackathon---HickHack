// robot-router Worker: voice/text instruction -> Workers AI -> one of the robot's trained skills.
//
//   GET  /api/agent          WebSocket for the laptop agent (host/robot_agent.py)
//   GET  /api/state          robot + last command / result
//   GET  /api/image/:kind    before | after snapshot (JPEG)
//   POST /api/command        {text} or {audio: base64 wav}  -> transcribe, look, route, guard, run
//   POST /api/run            {skill}   run a skill directly (confirm button / bypass when AI is down)
//   POST /api/stop           stop the running skill
// All /api routes need the shared secret: "Authorization: Bearer <ROBOT_TOKEN>" (agent: ?token=...).

import { describeScene, route, transcribe } from "./ai";
import { RobotSession, type CommandLog, type Env } from "./session";

export { RobotSession };

const CONFIDENCE_MIN = 0.6;

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

    if (url.pathname === "/api/agent") return robot.fetch(req);
    if (url.pathname === "/api/state" && req.method === "GET") return json(await robot.getState());

    const img = url.pathname.match(/^\/api\/image\/(before|after)$/);
    if (img && req.method === "GET") {
      const b64 = await robot.getImage(img[1] as "before" | "after");
      if (!b64) return new Response("no image", { status: 404 });
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      return new Response(bytes, { headers: { "content-type": "image/jpeg", "cache-control": "no-store" } });
    }

    if (req.method !== "POST") return json({ error: "not found" }, 404);
    const body = (await req.json().catch(() => ({}))) as { text?: string; audio?: string; skill?: string };

    if (url.pathname === "/api/stop") return json({ sent: await robot.stop() });

    if (url.pathname === "/api/run") {
      if (!body.skill) return json({ error: "skill required" }, 400);
      return json(await robot.run(body.skill));
    }

    if (url.pathname === "/api/command") return json(await handleCommand(env, robot, body));

    return json({ error: "not found" }, 404);
  },
} satisfies ExportedHandler<Env>;

async function handleCommand(
  env: Env,
  robot: DurableObjectStub<RobotSession>,
  body: { text?: string; audio?: string },
): Promise<CommandLog & { snapshot: boolean }> {
  const log: CommandLog = { at: new Date().toISOString(), transcript: "", scene: "", decision: null, action: "error" };
  let snapshot = false;
  try {
    // 1. what was said
    log.transcript = body.audio ? await transcribe(env.AI, body.audio) : (body.text ?? "").trim();
    if (!log.transcript) throw new Error("empty instruction");

    const state = await robot.getState();
    if (!state.agentConnected) {
      log.action = "no_agent";
      return { ...log, snapshot };
    }

    // 2. what the camera sees (one frame, same camera the policy uses)
    const jpeg = await robot.snapshot(5000);
    snapshot = !!jpeg;
    if (jpeg) log.scene = await describeScene(env.AI, jpeg).catch((e) => `(vision failed: ${e})`);

    // 3. which skill - the LLM can only pick from the agent's list
    log.decision = await route(env.AI, log.transcript, state.skills, log.scene);

    // 4. guardrails, in code rather than in the prompt
    if (log.decision.policy === "none") log.action = "none";
    else if (log.decision.confidence < CONFIDENCE_MIN) log.action = "needs_confirm";
    else {
      const r = await robot.run(log.decision.policy);
      log.action = r.ok ? "started" : r.reason.startsWith("busy") ? "busy" : "error";
      if (!r.ok) log.error = r.reason;
    }
  } catch (e) {
    log.action = "error";
    log.error = String(e);
  }
  await robot.logCommand(log);
  return { ...log, snapshot };
}
