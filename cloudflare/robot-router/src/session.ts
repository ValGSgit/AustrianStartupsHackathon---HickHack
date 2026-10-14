// RobotSession: one Durable Object instance per robot. Holds the laptop agent's WebSocket
// (hibernation API, so an idle connection costs nothing) and the robot's state.
//
// Agent -> DO messages:   hello {skills, mode} | snapshot {id, jpeg} | status {state, detail} | done {skill, ok, detail, jpeg?}
// DO -> agent messages:   snapshot {id} | run {skill} | stop

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
  transcript: string;
  scene: string;
  decision: Decision | null;
  action: string; // started | needs_confirm | none | busy | no_agent | error
  error?: string;
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
  hasBefore: boolean;
  hasAfter: boolean;
}

type Pending = { resolve: (jpeg: string | null) => void; timer: ReturnType<typeof setTimeout> };

export class RobotSession extends DurableObject<Env> {
  private pending = new Map<string, Pending>();

  // ------------------------------------------------------------ agent WebSocket

  async fetch(request: Request): Promise<Response> {
    if (request.headers.get("Upgrade") !== "websocket") return new Response("expected websocket", { status: 426 });
    for (const old of this.ctx.getWebSockets("agent")) old.close(1000, "replaced by a new agent connection");
    const pair = new WebSocketPair();
    this.ctx.acceptWebSocket(pair[1], ["agent"]);
    await this.patch({ status: "idle", detail: "agent connected", current: null });
    return new Response(null, { status: 101, webSocket: pair[0] });
  }

  async webSocketMessage(ws: WebSocket, raw: string | ArrayBuffer): Promise<void> {
    let msg: any;
    try {
      msg = JSON.parse(typeof raw === "string" ? raw : new TextDecoder().decode(raw));
    } catch {
      return;
    }
    switch (msg.type) {
      case "hello":
        await this.patch({
          skills: (msg.skills ?? []).map((s: Skill) => ({ name: String(s.name), description: String(s.description) })),
          mode: String(msg.mode ?? "real"),
          status: "idle",
          detail: `agent ready (${msg.mode ?? "real"})`,
        });
        break;
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
      case "done": {
        const skill = (await this.state()).skills.find((s) => s.name === msg.skill) ?? {
          name: String(msg.skill),
          description: String(msg.skill),
        };
        await this.patch({ status: "idle", current: null, detail: String(msg.detail ?? "done") });
        const result = { skill: skill.name, ok: !!msg.ok, detail: String(msg.detail ?? ""), verdict: "", at: new Date().toISOString() };
        await this.ctx.storage.put("lastResult", result);
        if (msg.jpeg) {
          await this.ctx.storage.put("img:after", msg.jpeg);
          // Ask the vision model whether it worked; don't block the agent on it.
          this.ctx.waitUntil(
            verifyOutcome(this.env.AI, msg.jpeg, skill)
              .then((verdict) => this.ctx.storage.put("lastResult", { ...result, verdict }))
              .catch((e) => this.ctx.storage.put("lastResult", { ...result, verdict: `check failed: ${e}` })),
          );
        }
        break;
      }
    }
  }

  async webSocketClose(): Promise<void> {
    if (this.ctx.getWebSockets("agent").length <= 1) {
      await this.patch({ status: "offline", detail: "agent disconnected", current: null });
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
    if (jpeg) await this.ctx.storage.put("img:before", jpeg);
    return jpeg;
  }

  async run(skill: string): Promise<{ ok: boolean; reason: string }> {
    const s = await this.state();
    const ws = this.agent();
    if (!ws) return { ok: false, reason: "robot agent is not connected" };
    if (!s.skills.some((k) => k.name === skill)) return { ok: false, reason: `unknown skill '${skill}'` };
    if (s.status === "running") return { ok: false, reason: `busy running '${s.current}'` };
    ws.send(JSON.stringify({ type: "run", skill }));
    await this.ctx.storage.delete("img:after");
    await this.patch({ status: "running", current: skill, detail: `running ${skill}` });
    return { ok: true, reason: "started" };
  }

  async stop(): Promise<boolean> {
    const ws = this.agent();
    ws?.send(JSON.stringify({ type: "stop" }));
    await this.patch({ detail: "stop requested" });
    return !!ws;
  }

  async logCommand(entry: CommandLog): Promise<void> {
    await this.ctx.storage.put("lastCommand", entry);
  }

  // ------------------------------------------------------------ helpers

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
      hasBefore: !!(await this.ctx.storage.get("img:before")),
      hasAfter: !!(await this.ctx.storage.get("img:after")),
    };
  }

  private async patch(p: Partial<State>): Promise<void> {
    const base = (await this.ctx.storage.get<Partial<State>>("state")) ?? {};
    await this.ctx.storage.put("state", { ...base, ...p });
  }
}
