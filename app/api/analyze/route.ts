import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

import { guardApiRequest, safeErrorResponse } from "@/lib/api-guard";

import {
  PROFILE_ROUTER_PROMPT,
  buildProfiledAnalysisPrompt,
  normalizeItemProfile,
} from "@/lib/prompts";

import { optimizeTitle } from "@/lib/titleOptimizer";
import { applyPriceMarkup, priceMarkupPercent } from "@/lib/pricing";
import type { AnalyzeRequestBody, ListingResult } from "@/lib/types";

export const maxDuration = 300;

const ANALYSIS_MODEL = "gpt-4.1-mini";
const ROUTER_MODEL = "gpt-4.1-mini";

const MAX_IMAGES = 12;

const ANALYZE_TIME_BUDGET_MS = 250_000;
const ROUTER_TIMEOUT_MS = 20_000;
const ANALYSIS_CALL_TIMEOUT_MS = 120_000;
const MIN_CALL_MS = 5_000;

type OpenAIImagePart = {
  type: "image_url";
  image_url: {
    url: string;
  };
};

type OpenAITextPart = {
  type: "text";
  text: string;
};

type OpenAIContentPart = OpenAIImagePart | OpenAITextPart;

function getClient(): OpenAI {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is not set. Add it in Vercel → Settings → Environment Variables."
    );
  }

  return new OpenAI({
    apiKey,
    maxRetries: 0,
  });
}

function rawBase64(data: string): string {
  return data.includes(",") ? data.split(",")[1] : data;
}

function toImageParts(
  images: AnalyzeRequestBody["images"]
): OpenAIImagePart[] {
  const parts: OpenAIImagePart[] = [];

  for (const img of images.slice(0, MAX_IMAGES)) {
    if (!img?.data || !img?.mediaType) continue;

    const allowed = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ].includes(img.mediaType);

    if (!allowed) continue;

    const data = rawBase64(img.data);

    parts.push({
      type: "image_url",
      image_url: {
        url: `data:${img.mediaType};base64,${data}`,
      },
    });
  }

  return parts;
}

function parseModelJson<T>(raw: string): T {
  let text = raw.trim();

  if (text.startsWith("```")) {
    text = text
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/, "")
      .trim();
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    const match = text.match(/\{[\s\S]*\}/);

    if (match) {
      return JSON.parse(match[0]) as T;
    }

    throw new Error("OpenAI did not return valid JSON.");
  }
}

async function routeProfile(
  client: OpenAI,
  imageParts: OpenAIImagePart[],
  requested: string,
  routerModel: string,
  deadline: number
): Promise<string> {
  const forced = normalizeItemProfile(requested);

  if (forced !== "auto") {
    return forced;
  }

  const remaining = deadline - Date.now();

  if (remaining < MIN_CALL_MS) {
    return "hard_goods";
  }

  try {
    const content: OpenAIContentPart[] = [
      ...imageParts,
      {
        type: "text",
        text: PROFILE_ROUTER_PROMPT,
      },
    ];

    const controller = new AbortController();

    const timeout = setTimeout(() => {
      controller.abort();
    }, Math.min(ROUTER_TIMEOUT_MS, remaining));

    try {
      const response = await client.chat.completions.create(
        {
          model: routerModel,
          max_tokens: 300,
          response_format: {
            type: "json_object",
          },
          messages: [
            {
              role: "user",
              content,
            },
          ],
        },
        {
          signal: controller.signal,
        }
      );

      const text = response.choices[0]?.message?.content ?? "";

      const data = parseModelJson<{ profile?: string }>(text);

      const routed = normalizeItemProfile(data?.profile ?? "auto");

      return routed !== "auto" ? routed : "hard_goods";
    } finally {
      clearTimeout(timeout);
    }
  } catch {
    return "hard_goods";
  }
}

export async function POST(req: NextRequest) {
  const denied = guardApiRequest(req);

  if (denied) {
    return denied;
  }

  let body: AnalyzeRequestBody;

  try {
    body = (await req.json()) as AnalyzeRequestBody;
  } catch {
    return NextResponse.json(
      {
        ok: false,
        error: "Invalid request body.",
      },
      {
        status: 400,
      }
    );
  }

  if (!Array.isArray(body.images) || body.images.length === 0) {
    return NextResponse.json(
      {
        ok: false,
        error: "Please add at least one photo.",
      },
      {
        status: 400,
      }
    );
  }

  const imageParts = toImageParts(body.images);

  if (imageParts.length === 0) {
    return NextResponse.json(
      {
        ok: false,
        error: "No readable photos found. Use JPG, PNG, or WebP.",
      },
      {
        status: 400,
      }
    );
  }

  let client: OpenAI;

  try {
    client = getClient();
  } catch (e) {
    return NextResponse.json(
      {
        ok: false,
        error: e instanceof Error ? e.message : "OpenAI configuration error.",
      },
      {
        status: 500,
      }
    );
  }

  try {
    const deadline = Date.now() + ANALYZE_TIME_BUDGET_MS;

    const profile = await routeProfile(
      client,
      imageParts,
      body.profile,
      ROUTER_MODEL,
      deadline
    );

    const systemPrompt = buildProfiledAnalysisPrompt(profile);

    let lastErr: unknown = null;

    for (let attempt = 0; attempt < 3; attempt++) {
      const remaining = deadline - Date.now();

      if (remaining < MIN_CALL_MS) {
        lastErr =
          lastErr ??
          new Error(
            "Analysis ran out of time — the OpenAI API is slow right now."
          );

        break;
      }

      try {
        const content: OpenAIContentPart[] = [
          ...imageParts,
          {
            type: "text",
            text: "Analyze these photos and return the listing JSON now.",
          },
        ];

        const controller = new AbortController();

        const timeout = setTimeout(() => {
          controller.abort();
        }, Math.min(ANALYSIS_CALL_TIMEOUT_MS, remaining));

        try {
          const response = await client.chat.completions.create(
            {
              model: ANALYSIS_MODEL,
              max_tokens: 3000,
              response_format: {
                type: "json_object",
              },
              messages: [
                {
                  role: "system",
                  content: systemPrompt,
                },
                {
                  role: "user",
                  content,
                },
              ],
            },
            {
              signal: controller.signal,
            }
          );

          const text = response.choices[0]?.message?.content ?? "";

          const listing = parseModelJson<ListingResult>(text);

          listing.item_profile = profile;

          listing.title = optimizeTitle(listing);

          listing.suggested_price = applyPriceMarkup(
            listing.suggested_price,
            priceMarkupPercent()
          );

          return NextResponse.json({
            ok: true,
            listing,
          });
        } finally {
          clearTimeout(timeout);
        }
      } catch (err) {
        lastErr = err;

        if (attempt < 2) {
          await new Promise((resolve) =>
            setTimeout(resolve, 1500 * (attempt + 1))
          );
        }
      }
    }

    throw lastErr ?? new Error("OpenAI analysis failed.");
  } catch (e) {
    console.error("[analyze] OpenAI failure:", e);

    return safeErrorResponse(
      "analyze",
      e,
      "Something went wrong analyzing photos — please try again."
    );
  }
}
