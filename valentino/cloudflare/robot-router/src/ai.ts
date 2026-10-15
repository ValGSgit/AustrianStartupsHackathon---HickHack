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
  instruction?: boolean; // the policy reads a task text (SmolVLA): the router writes it
}

export interface Decision {
  policy: string; // a skill name, or "none"
  confidence: number;
  say: string;
  task: string; // instruction for a skill that takes one, e.g. "Pick up the blue pen"; "" otherwise
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
    skills.map((s) => `- ${s.name}: ${s.description}${s.instruction ? " [takes an instruction]" : ""}`).join("\n") +
    `\nCamera sees: ${scene || "(no camera snapshot available)"}\n` +
    "Pick exactly one skill that does what the user asks, or 'none' if no skill matches or the object it " +
    "needs is clearly missing. confidence is 0..1. 'say' is one short friendly sentence for the audience " +
    "describing what the robot will do (or why it won't). 'task': for a skill marked [takes an instruction], " +
    "the user's request as one short imperative sentence for the robot's model, naming the object and its " +
    "color if known (e.g. 'Pick up the blue pen'); otherwise an empty string.";

  const schema = {
    type: "object",
    properties: {
      policy: { type: "string", enum: [...names, "none"] },
      confidence: { type: "number" },
      say: { type: "string" },
      task: { type: "string" },
    },
    required: ["policy", "confidence", "say", "task"],
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
        task: String(d.task ?? "").trim(),
      };
    } catch (e) {
      lastError = e; // JSON mode can fail ("JSON Mode couldn't be met") - retry once
    }
  }
  throw lastError;
}

// ---------------------------------------------------------------------------------------------
// Intent check: "move the orange disc to box B" -> which object, which drawn box, how sure.
// Same prompt and rules as host/intent_test.py (tested there on the real camera).

export const BOXES = ["A", "B", "C", "D", "X", "Z"];

export interface IntentAnswer {
  object?: string;
  object_where?: string;
  target_box?: string;
  target_where?: string;
  boxes_seen?: string[];
  confidence?: number;
  problem?: string;
}

export interface Intent {
  answer: IntentAnswer; // the temperature-0 answer: what the robot would act on
  box: string;
  verdict: "GO" | "ASK";
  reasons: string[]; // why ASK
  agreement: number; // 0..1, same object AND same box across all answers
  boxAgreement: number;
  objectAgreement: number;
  selfConfidence: number | null; // the model's own 0..100 (usually over-confident; shown for comparison)
  boxVotes: Record<string, number>;
  objectVotes: Record<string, number>;
  answers: IntentAnswer[];
  ms: number;
  model: string;
}

const intentPrompt = (instruction: string, boxes: string[]) =>
  `You are the eyes of a robot arm. The image is from its camera, looking at a table.
On the table there is paper with boxes (rectangles) drawn on it; each box has one letter inside: ${boxes.join(", ")}.
Some boxes may be partly outside the picture or seen at a steep angle.

The user says: "${instruction.replace(/"/g, "'")}"

Work out what the user wants moved and into which lettered box. Answer with ONLY this JSON, no other text:
{"object": "short name of the thing to move, with its colour",
 "object_where": "where that object is now, in words (e.g. 'just above box D')",
 "target_box": "one of ${JSON.stringify(boxes)} or 'none' if the instruction names no box or that box isn't visible",
 "target_where": "where that box is in the picture, in words",
 "boxes_seen": [letters you can actually read in the image],
 "confidence": 0-100 how sure you are that you understood both the object and the box,
 "problem": "anything unclear (object not visible, two candidates, box not found) or empty"}`;

async function askIntent(ai: Ai, jpeg: string, prompt: string, temperature: number): Promise<IntentAnswer> {
  try {
    const out = (await ai.run(MODELS.vision, {
      messages: [{ role: "user", content: [
        { type: "text", text: prompt },
        { type: "image_url", image_url: { url: `data:image/jpeg;base64,${jpeg}` } },
      ] }],
      max_tokens: 300,
      temperature,
    })) as { response?: unknown };
    if (out.response && typeof out.response === "object") return out.response as IntentAnswer;
    const m = String(out.response ?? "").match(/\{[\s\S]*\}/); // models sometimes wrap the JSON in ``` or add a sentence
    return m ? (JSON.parse(m[0]) as IntentAnswer) : { problem: `no JSON in answer: ${String(out.response).slice(0, 120)}` };
  } catch (e) {
    return { problem: `call failed: ${e}` };
  }
}

const STOP = new Set(["the", "a", "an", "small", "little", "round", "flat", "object", "piece", "of", "thing", "big", "large"]);
const words = (s: unknown) => new Set((String(s ?? "").toLowerCase().match(/[a-z]+/g) ?? []).filter((w) => !STOP.has(w)));
const boxOf = (a: IntentAnswer) => String(a.target_box ?? "none").trim().toUpperCase().replace(/^BOX\s+/, "");

/** runs answers in parallel: the first at temperature 0 (the one the robot uses), the rest at 0.7 to measure stability. */
export async function inferIntent(ai: Ai, jpeg: string, instruction: string, runs = 5, boxes = BOXES): Promise<Intent> {
  const t0 = Date.now();
  const prompt = intentPrompt(instruction, boxes);
  const answers = await Promise.all(Array.from({ length: runs }, (_, i) => askIntent(ai, jpeg, prompt, i === 0 ? 0 : 0.7)));
  const best = answers[0];
  const box = boxOf(best);
  const ref = words(best.object);
  // same object = one name's words contain the other's ("orange eraser" ~ "orange"); "red pen" vs "blue pen" don't
  const sameObj = answers.map((a) => {
    const w = words(a.object);
    return ref.size > 0 && w.size > 0 && ([...ref].every((x) => w.has(x)) || [...w].every((x) => ref.has(x)));
  });
  const sameBox = answers.map((a) => boxOf(a) === box);
  const count = (xs: string[]) => xs.reduce<Record<string, number>>((m, x) => ((m[x] = (m[x] ?? 0) + 1), m), {});
  const agreement = sameObj.filter((o, i) => o && sameBox[i]).length / runs;
  const seen = (Array.isArray(best.boxes_seen) ? best.boxes_seen : []).map((b) => String(b).trim().toUpperCase());
  const self = answers.map((a) => Number(a.confidence)).filter((c) => Number.isFinite(c));
  // the model guesses even when unsure but writes the doubt into 'problem': gate on facts, not on its own score
  const reasons = [
    agreement < 0.8 ? `answers disagree (${Math.round(agreement * 100)}% < 80%)` : "",
    !seen.includes(box) ? `box ${box} is not one it could read (${seen.join(", ") || "none"})` : "",
    String(best.problem ?? "").trim() ? `model reports: ${best.problem}` : "",
  ].filter(Boolean);
  return {
    answer: best, box, verdict: reasons.length ? "ASK" : "GO", reasons, agreement,
    boxAgreement: sameBox.filter(Boolean).length / runs,
    objectAgreement: sameObj.filter(Boolean).length / runs,
    selfConfidence: self.length ? self.reduce((a, b) => a + b, 0) / self.length : null,
    boxVotes: count(answers.map(boxOf)),
    objectVotes: count(answers.map((a) => [...words(a.object)].join(" ") || "?")),
    answers, ms: Date.now() - t0, model: MODELS.vision,
  };
}
