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

/** Tells the vision model what the arm is; the sim's yellow arm is otherwise read as a "vehicle". */
const arm = (mode: string) =>
  `Front camera of a robot workspace. The large ${mode === "sim" ? "yellow " : ""}object in the middle is the robot ` +
  "arm itself (an SO-101 arm with a two-finger gripper), standing on a table. ";

export async function describeScene(ai: Ai, jpegBase64: string, mode: string): Promise<string> {
  const out = (await ai.run(MODELS.vision, {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            // Tested on the sim camera: without being told what the arm is, Mistral calls it a "vehicle", and a
            // "list objects or say 'no objects'" prompt makes it answer 'no objects' even with a pen in view.
            text:
              arm(mode) +
              "Describe in one sentence what else is on the table besides the arm: name each object with " +
              "its color, what it probably is (pen, cube, cup, ...) and where it is relative to the gripper. " +
              "If nothing else is there, say 'nothing besides the arm'. No intro.",
          },
          { type: "image_url", image_url: { url: `data:image/jpeg;base64,${jpegBase64}` } },
        ],
      },
    ],
    max_tokens: 120,
  })) as { response?: string };
  return (out.response ?? "").trim();
}

export async function verifyOutcome(ai: Ai, jpegBase64: string, skill: Skill, mode: string): Promise<string> {
  const out = (await ai.run(MODELS.vision, {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
              arm(mode) +
              `It just tried to do this: "${skill.description}". This is one still image taken right after. ` +
              `If that task is only a motion that leaves nothing to see afterwards (like waving), answer ` +
              `"n/a - motion only". Otherwise answer "yes" or "no", then at most 12 words why.`,
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
