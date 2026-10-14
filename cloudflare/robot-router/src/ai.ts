// Workers AI calls: speech-to-text, scene description, instruction -> skill routing, success check.
// Model ids checked against the Workers AI catalog / @cloudflare/workers-types on 2026-10-14.

export const MODELS = {
  stt: "@cf/openai/whisper-large-v3-turbo",
  // Mistral Small 3.1 (Apache-2.0). Not Llama 3.2 Vision: its license requires confirming you are NOT
  // an EU-based individual/company, which rules it out for an Austrian team.
  vision: "@cf/mistralai/mistral-small-3.1-24b-instruct",
  router: "@cf/meta/llama-3.3-70b-instruct-fp8-fast", // supports JSON mode
} as const;

export interface Skill {
  name: string;
  description: string;
}

export interface Decision {
  policy: string; // a skill name, or "none"
  confidence: number;
  say: string;
}

export async function transcribe(ai: Ai, audioBase64: string): Promise<string> {
  const out = await ai.run(MODELS.stt, { audio: audioBase64, language: "en", vad_filter: true });
  return (out.text ?? "").trim();
}

export async function describeScene(ai: Ai, jpegBase64: string): Promise<string> {
  const out = (await ai.run(MODELS.vision, {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
              "Camera view of a robot arm's workspace. IGNORE the robot arm itself, its base and the table/mat. " +
              "List every small loose object lying on the table, e.g. pens, markers, cubes, blocks, cups " +
              "(color + type + position left/center/right). Look carefully, objects can be thin. " +
              "If there are none, say 'no objects'. One short line, no intro.",
          },
          { type: "image_url", image_url: { url: `data:image/jpeg;base64,${jpegBase64}` } },
        ],
      },
    ],
    max_tokens: 120,
  })) as { response?: string };
  return (out.response ?? "").trim();
}

export async function verifyOutcome(ai: Ai, jpegBase64: string, skill: Skill): Promise<string> {
  const out = (await ai.run(MODELS.vision, {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
              `A robot arm just tried to do this: "${skill.description}". Looking at this image, did it ` +
              `succeed? Answer "yes" or "no", then at most 12 words why.`,
          },
          { type: "image_url", image_url: { url: `data:image/jpeg;base64,${jpegBase64}` } },
        ],
      },
    ],
    max_tokens: 40,
  })) as { response?: string };
  return (out.response ?? "").trim();
}

/** Pick exactly one trained skill (or "none") for the instruction. The enum makes anything else impossible. */
export async function route(ai: Ai, instruction: string, skills: Skill[], scene: string): Promise<Decision> {
  const names = skills.map((s) => s.name);
  const system =
    "You control a robot arm that can ONLY run these skills:\n" +
    skills.map((s) => `- ${s.name}: ${s.description}`).join("\n") +
    `\nCamera sees: ${scene || "(no camera snapshot available)"}\n` +
    "Pick exactly one skill that does what the user asks, or 'none' if no skill matches or the object it " +
    "needs is clearly missing. confidence is 0..1. 'say' is one short friendly sentence for the audience " +
    "describing what the robot will do (or why it won't).";

  const schema = {
    type: "object",
    properties: {
      policy: { type: "string", enum: [...names, "none"] },
      confidence: { type: "number" },
      say: { type: "string" },
    },
    required: ["policy", "confidence", "say"],
  };

  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const out = (await ai.run(MODELS.router, {
        messages: [
          { role: "system", content: system },
          { role: "user", content: instruction },
        ],
        response_format: { type: "json_schema", json_schema: schema },
        max_tokens: 200,
        temperature: 0,
      })) as { response?: unknown };
      const r = typeof out.response === "string" ? JSON.parse(out.response) : out.response;
      const d = r as Decision;
      if (typeof d?.policy !== "string") throw new Error("router returned no policy");
      return {
        policy: [...names, "none"].includes(d.policy) ? d.policy : "none",
        confidence: Math.max(0, Math.min(1, Number(d.confidence) || 0)),
        say: String(d.say ?? ""),
      };
    } catch (e) {
      lastError = e; // JSON mode can fail ("JSON Mode couldn't be met") - retry once
    }
  }
  throw lastError;
}
