import OpenAI from "openai";

let client: OpenAI | null = null;

export function getOpenAIClient(): OpenAI {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is not set. Add it in Vercel → Settings → Environment Variables."
    );
  }

  if (!client) {
    client = new OpenAI({ apiKey });
  }

  return client;
}

/**
 * Compatibility wrapper.
 *
 * Existing HAHM code expects:
 *
 *   getClient().messages.create(...)
 *
 * We keep that interface temporarily so we do NOT have to rewrite
 * every file in the application at once.
 */
export function getClient() {
  const openai = getOpenAIClient();

  return {
    messages: {
      async create(params: {
        model: string;
        max_tokens?: number;
        messages: Array<{
          role: "user" | "assistant" | "system";
          content: string | unknown[];
        }>;
      }) {
        const model =
          process.env.OPENAI_MODEL ||
          "gpt-5.5";

        const messages = params.messages.map((message) => ({
          role: message.role,
          content:
            typeof message.content === "string"
              ? message.content
              : JSON.stringify(message.content),
        }));

        const response = await openai.chat.completions.create({
          model,
          max_tokens: params.max_tokens ?? 1500,
          messages,
        });

        const text = response.choices[0]?.message?.content ?? "";

        return {
          content: [
            {
              type: "text" as const,
              text,
            },
          ],
        };
      },
    },
  };
}

/**
 * Parse JSON returned by the model.
 *
 * Handles normal JSON as well as JSON wrapped in markdown fences.
 */
export function parseModelJson<T = unknown>(raw: string): T {
  let text = raw.trim();

  text = text
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    return JSON.parse(text) as T;
  } catch {
    const match = text.match(/\{[\s\S]*\}|\[[\s\S]*\]/);

    if (match) {
      try {
        return JSON.parse(match[0]) as T;
      } catch {
        // Continue to the final error.
      }
    }

    throw new Error("Model did not return valid JSON.");
  }
}

/**
 * Compatibility error class.
 *
 * Existing routes may import this name, so keep it for now.
 */
export class AnthropicAuthError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "AnthropicAuthError";
    this.status = status;
  }
}

/**
 * Convert API errors into the error type expected by
 * the existing application.
 */
export function anthropicAuthError(
  error: unknown
): AnthropicAuthError | null {
  const status =
    typeof error === "object" &&
    error !== null &&
    "status" in error &&
    typeof (error as { status?: unknown }).status === "number"
      ? (error as { status: number }).status
      : undefined;

  const message =
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof (error as { message?: unknown }).message === "string"
      ? (error as { message: string }).message
      : error instanceof Error
        ? error.message
        : "";

  if (status === 401) {
    return new AnthropicAuthError(
      "OpenAI API key was rejected (401). Check OPENAI_API_KEY in your environment variables.",
      401
    );
  }

  if (status === 403) {
    return new AnthropicAuthError(
      "OpenAI API access was denied (403). Check the API key and project permissions.",
      403
    );
  }

  if (
    status === 402 ||
    status === 429 ||
    /billing|quota|insufficient/i.test(message)
  ) {
    return new AnthropicAuthError(
      "OpenAI API request was rejected because of billing, quota, or rate limits.",
      status ?? 429
    );
  }

  return null;
}
