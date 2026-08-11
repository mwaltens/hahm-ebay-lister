import { NextRequest, NextResponse } from "next/server";

import OpenAI from "openai";

import {
  buildVerifyMergePrompt,
} from "@/lib/prompts";

import type { WireImage } from "@/lib/images";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const MODEL =
  process.env.OPENAI_CHECK_MODEL ||
  process.env.OPENAI_MODEL ||
  "gpt-5.5";

type MergeCheckRequest = {
  imageA: WireImage;
  imageB: WireImage;
  countA?: number;
  countB?: number;
};

type MergeCheckResponse = {
  merge: boolean;
};

function getImageUrl(image: WireImage): string {
  const raw = image as unknown as Record<string, unknown>;

  const value =
    typeof raw.url === "string"
      ? raw.url
      : typeof raw.imageUrl === "string"
        ? raw.imageUrl
        : typeof raw.image_url === "string"
          ? raw.image_url
          : typeof raw.dataUrl === "string"
            ? raw.dataUrl
            : typeof raw.dataURL === "string"
              ? raw.dataURL
              : typeof raw.src === "string"
                ? raw.src
                : typeof raw.data === "string"
                  ? raw.data
                  : typeof raw.base64 === "string"
                    ? raw.base64
                    : null;

  if (!value) {
    throw new Error(
      "Image does not contain a usable URL or base64 data.",
    );
  }

  if (
    value.startsWith("http://") ||
    value.startsWith("https://") ||
    value.startsWith("data:")
  ) {
    return value;
  }

  const mimeType =
    typeof raw.mimeType === "string"
      ? raw.mimeType
      : typeof raw.contentType === "string"
        ? raw.contentType
        : "image/jpeg";

  return `data:${mimeType};base64,${value}`;
}

function parseMergeResponse(text: string): MergeCheckResponse {
  let cleaned = text.trim();

  cleaned = cleaned
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    const parsed = JSON.parse(cleaned);

    return {
      merge: parsed.merge === true,
    };
  } catch {
    const match = cleaned.match(/\{[\s\S]*\}/);

    if (match) {
      try {
        const parsed = JSON.parse(match[0]);

        return {
          merge: parsed.merge === true,
        };
      } catch {
        // Fall through.
      }
    }
  }

  throw new Error(
    "OpenAI returned invalid merge-check JSON.",
  );
}

export async function POST(request: NextRequest) {
  try {
    const body =
      (await request.json()) as MergeCheckRequest;

    if (!body.imageA || !body.imageB) {
      return NextResponse.json(
        {
          error:
            "imageA and imageB are required.",
        },
        { status: 400 },
      );
    }

    const prompt = buildVerifyMergePrompt(
      body.countA ?? 1,
      body.countB ?? 1,
    );

    const response = await openai.responses.create({
      model: MODEL,

      input: [
        {
          role: "user",

          content: [
            {
              type: "input_text",
              text: prompt,
            },

            {
              type: "input_text",
              text: "Group A / first image:",
            },

            {
              type: "input_image",
              image_url: getImageUrl(body.imageA),
            },

            {
              type: "input_text",
              text: "Group B / second image:",
            },

            {
              type: "input_image",
              image_url: getImageUrl(body.imageB),
            },
          ],
        },
      ],

      text: {
        format: {
          type: "json_object",
        },
      },

      max_output_tokens: 300,

      store: false,
    });

    const result = parseMergeResponse(
      response.output_text,
    );

    return NextResponse.json(result);
  } catch (error) {
    console.error(
      "[merge-check] OpenAI error:",
      error,
    );

    if (error instanceof OpenAI.APIError) {
      return NextResponse.json(
        {
          error:
            error.message ||
            "OpenAI request failed.",
        },
        {
          status: error.status || 500,
        },
      );
    }

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Merge check failed.",
      },
      { status: 500 },
    );
  }
}
