// RobotSession: one Durable Object instance per robot. Holds the laptop agent's WebSocket
// (hibernation API, so an idle connection costs nothing) and the robot's state.
//
// Agent -> DO messages:   hello {skills, mode, arm, camera} | snapshot {id, jpeg} | status {state, detail} | done {skill, ok, detail, jpeg?}
//                         hw {arm, camera}   (device status for the dashboard's status line)
//                         log {text} | frame {t, jpeg, joints, state, target}   (live view while a skill runs)
// DO -> agent messages:   snapshot {id} | run {skill} | stop | hw {action: camera_on|camera_off|arm_on|arm_off}
// DO -> viewer messages:  backlog {events, frame} | event {event} | frame {...}   (dashboard Live panel, /api/live)

import { DurableObject } from "cloudflare:workers";
import { verifyOutcome, type Decision, type Skill } from "./ai";

export interface Env {
  AI: Ai;
  ROBOT: DurableObjectNamespace<RobotSession>;
  ASSETS: Fetcher;
  ROBOT_TOKEN: string;
}

export interface CommandLog {
  at: string;
  source?: "voice" | "text" | "button"; // button = skill run directly from the dashboard, no AI
  transcript: string;
  scene: string;
  decision: Decision | null;
  action: string; // started | needs_confirm | none | busy | no_agent | error
  error?: string;
}

export interface Device {
  state: string; // on | off | warn | error
  detail: string;
}

export interface State {
  agentConnected: boolean;
  mode: string; // "real" | "sim" (reported by the agent)
  skills: Skill[];
  status: string; // idle | running | error | offline
  current: string | null;
  detail: string;
  lastCommand: CommandLog | null;
  lastResult: { skill: string; ok: boolean; detail: string; verdict: string; at: string } | null;
  images: { before: string | null; after: string | null }; // when each snapshot was stored (dashboard reloads on change)
  hw: { arm: Device; camera: Device } | null;
}

const device = (d: any): Device => ({ state: String(d?.state ?? "off"), detail: String(d?.detail ?? "") });

type Pending = { resolve: (jpeg: string | null) => void; timer: ReturnType<typeof setTimeout> };

/** One line of the pipeline log the dashboard shows: heard -> camera -> vision -> router -> run -> inference -> result. */
export interface LiveEvent {
  at: string;
  kind: string; // heard | camera | vision | router | guard | run | agent | done | verify | stop | error | conn
  text: string;
}

const MAX_EVENTS = 120; // kept in storage, so a freshly opened dashboard still shows the last commands
const AI_PER_MINUTE = 12; // commands + looks that call Workers AI; protects the bill if the token leaks or a script loops

export class RobotSession extends DurableObject<Env> {
  private pending = new Map<string, Pending>();
  private events: LiveEvent[] = [];
  private lastFrame: string | null = null; // memory only: frames are for watching, not keeping
  private aiCalls: number[] = [];

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.blockConcurrencyWhile(async () => {
      this.events = (await ctx.storage.get<LiveEvent[]>("events")) ?? [];
    });
  }

  // ------------------------------------------------------------ WebSockets: agent (/api/agent), viewers (/api/live)

  async fetch(request: Request): Promise<Response> {
    if (request.headers.get("Upgrade") !== "websocket") return new Response("expected websocket", { status: 426 });
    const pair = new WebSocketPair();
    if (new URL(request.url).pathname === "/api/live") {
      this.ctx.acceptWebSocket(pair[1], ["viewer"]);
      pair[1].send(`{"type":"backlog","events":${JSON.stringify(this.events)},"frame":${this.lastFrame ?? "null"}}`);
      pair[1].send(JSON.stringify({ type: "state", state: await this.state() }));
      return new Response(null, { status: 101, webSocket: pair[0] });
    }
    for (const old of this.ctx.getWebSockets("agent")) old.close(1000, "replaced by a new agent connection");
    this.ctx.acceptWebSocket(pair[1], ["agent"]);
    await this.patch({ status: "idle", detail: "agent connected", current: null });
    this.event("conn", "robot agent connected");
    return new Response(null, { status: 101, webSocket: pair[0] });
  }

  async webSocketMessage(ws: WebSocket, raw: string | ArrayBuffer): Promise<void> {
    if (!this.ctx.getTags(ws).includes("agent")) return; // viewers only listen
    const text = typeof raw === "string" ? raw : new TextDecoder().decode(raw);
    let msg: any;
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    switch (msg.type) {
      case "hello":
        await this.patch({
          skills: (msg.skills ?? []).map((s: Skill) => ({
            name: String(s.name),
            description: String(s.description),
            instruction: !!s.instruction,
          })),
          mode: String(msg.mode ?? "real"),
          status: "idle",
          detail: `agent ready (${msg.mode ?? "real"})`,
          hw: { arm: device(msg.arm), camera: device(msg.camera) },
        });
        break;
      case "hw": {
        const hw = { arm: device(msg.arm), camera: device(msg.camera) };
        await this.patch({ hw });
        this.event("conn", `arm: ${hw.arm.state} (${hw.arm.detail}) · camera: ${hw.camera.state} (${hw.camera.detail})`);
        break;
      }
      case "snapshot": {
        const p = this.pending.get(msg.id);
        if (p) {
          clearTimeout(p.timer);
          this.pending.delete(msg.id);
          p.resolve(msg.jpeg ?? null);
        }
        break;
      }
      case "status":
        await this.patch({ status: String(msg.state), detail: String(msg.detail ?? "") });
        break;
      case "log":
        this.event("agent", String(msg.text ?? ""));
        break;
      case "frame":
        this.lastFrame = text;
        this.broadcast(text);
        break;
      case "done": {
        const st = await this.state();
        const skill = st.skills.find((s) => s.name === msg.skill) ?? {
          name: String(msg.skill),
          description: String(msg.skill),
        };
        const result = { skill: skill.name, ok: !!msg.ok, detail: String(msg.detail ?? ""), verdict: "", at: new Date().toISOString() };
        await this.ctx.storage.put("lastResult", result);
        if (msg.jpeg) await this.putImage("after", msg.jpeg);
        await this.patch({ status: "idle", current: null, detail: String(msg.detail ?? "done") });
        this.event("done", `${skill.name}: ${result.ok ? "ok" : "not ok"} - ${result.detail}`);
        if (msg.jpeg) {
          // Ask the vision model whether it worked; don't block the agent on it.
          this.event("verify", "after-image sent to Mistral Small 3.1 to double-check the result…");
          this.ctx.waitUntil(
            verifyOutcome(this.env.AI, msg.jpeg, skill, st.mode)
              .catch((e) => `check failed: ${e}`)
              .then(async (verdict) => {
                this.event("verify", verdict);
                await this.ctx.storage.put("lastResult", { ...result, verdict });
                await this.pushState();
              }),
          );
        }
        break;
      }
    }
  }

  async webSocketClose(ws: WebSocket): Promise<void> {
    if (!this.ctx.getTags(ws).includes("agent")) return;
    if (this.ctx.getWebSockets("agent").length <= 1) {
      await this.patch({ status: "offline", detail: "agent disconnected", current: null });
      this.event("conn", "robot agent disconnected");
    }
  }

  // ------------------------------------------------------------ RPC used by the Worker

  async getState(): Promise<State> {
    return this.state();
  }

  async getImage(kind: "before" | "after"): Promise<string | null> {
    return (await this.ctx.storage.get<string>(`img:${kind}`)) ?? null;
  }

  /** Ask the agent for one camera frame (JPEG, base64). null if no agent or it doesn't answer in time. */
  async snapshot(timeoutMs = 5000): Promise<string | null> {
    const ws = this.agent();
    if (!ws) return null;
    const id = crypto.randomUUID();
    const jpeg = await new Promise<string | null>((resolve) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        resolve(null);
      }, timeoutMs);
      this.pending.set(id, { resolve, timer });
      ws.send(JSON.stringify({ type: "snapshot", id }));
    });
    if (jpeg) {
      await this.putImage("before", jpeg);
      await this.pushState();
    }
    return jpeg;
  }

  /** Rate limit for anything that calls Workers AI. */
  async allowAi(): Promise<boolean> {
    const now = Date.now();
    this.aiCalls = this.aiCalls.filter((t) => now - t < 60_000);
    if (this.aiCalls.length >= AI_PER_MINUTE) return false;
    this.aiCalls.push(now);
    return true;
  }

  async run(skill: string, task = ""): Promise<{ ok: boolean; reason: string }> {
    const s = await this.state();
    const ws = this.agent();
    if (!ws) return { ok: false, reason: "robot agent is not connected" };
    if (!s.skills.some((k) => k.name === skill)) return { ok: false, reason: `unknown skill '${skill}'` };
    if (s.status === "running") return { ok: false, reason: `busy running '${s.current}'` };
    const text = s.skills.find((k) => k.name === skill)?.instruction ? task : "";
    ws.send(JSON.stringify({ type: "run", skill, task: text }));
    this.event("run", `sent '${skill}' to the ${s.mode} robot` + (text ? ` with the instruction "${text}"` : ""));
    await this.ctx.storage.delete("img:after");
    await this.patch({ status: "running", current: skill, detail: `running ${skill}`, images: { ...s.images, after: null } });
    return { ok: true, reason: "started" };
  }

  async stop(): Promise<boolean> {
    const ws = this.agent();
    ws?.send(JSON.stringify({ type: "stop" }));
    this.event("stop", ws ? "STOP sent to the robot" : "STOP pressed, but no robot agent is connected");
    await this.patch({ detail: "stop requested" });
    return !!ws;
  }

  /** Camera preview / read-only arm connection on the laptop (status line buttons). */
  async hw(action: string): Promise<boolean> {
    const ws = this.agent();
    ws?.send(JSON.stringify({ type: "hw", action }));
    return !!ws;
  }

  async logCommand(entry: CommandLog): Promise<void> {
    await this.ctx.storage.put("lastCommand", entry);
    await this.pushState();
  }

  /** Append a line to the live log (called by the Worker at each pipeline stage). */
  async log(kind: string, text: string): Promise<void> {
    this.event(kind, text);
  }

  // ------------------------------------------------------------ helpers

  private event(kind: string, text: string): void {
    const e: LiveEvent = { at: new Date().toISOString(), kind, text };
    this.events.push(e);
    if (this.events.length > MAX_EVENTS) this.events.splice(0, this.events.length - MAX_EVENTS);
    this.ctx.storage.put("events", this.events); // no await: DO storage writes are ordered and coalesced
    this.broadcast(JSON.stringify({ type: "event", event: e }));
  }

  /** Dashboards get the robot state pushed on every change (they only poll as a fallback). */
  private async pushState(): Promise<void> {
    if (this.ctx.getWebSockets("viewer").length) this.broadcast(JSON.stringify({ type: "state", state: await this.state() }));
  }

  private async putImage(kind: "before" | "after", jpeg: string): Promise<void> {
    await this.ctx.storage.put(`img:${kind}`, jpeg);
    const base = (await this.ctx.storage.get<Partial<State>>("state")) ?? {};
    await this.ctx.storage.put("state", { ...base, images: { ...(base.images ?? { before: null, after: null }), [kind]: new Date().toISOString() } });
  }

  private broadcast(text: string): void {
    for (const v of this.ctx.getWebSockets("viewer")) {
      try {
        v.send(text);
      } catch {
        // viewer already gone; the runtime drops it
      }
    }
  }

  private agent(): WebSocket | null {
    return this.ctx.getWebSockets("agent")[0] ?? null;
  }

  private async state(): Promise<State> {
    const base = (await this.ctx.storage.get<Partial<State>>("state")) ?? {};
    return {
      agentConnected: !!this.agent(),
      mode: base.mode ?? "real",
      skills: base.skills ?? [],
      status: this.agent() ? (base.status ?? "idle") : "offline",
      current: base.current ?? null,
      detail: base.detail ?? "",
      lastCommand: (await this.ctx.storage.get<CommandLog>("lastCommand")) ?? null,
      lastResult: (await this.ctx.storage.get<State["lastResult"]>("lastResult")) ?? null,
      images: base.images ?? { before: null, after: null },
      hw: this.agent() ? (base.hw ?? null) : null,
    };
  }

  private async patch(p: Partial<State>): Promise<void> {
    const base = (await this.ctx.storage.get<Partial<State>>("state")) ?? {};
    await this.ctx.storage.put("state", { ...base, ...p });
    await this.pushState();
  }
}
