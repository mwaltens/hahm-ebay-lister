import OpenAI from "openai";

import {
  buildSortPrompt,
  buildVerifyGroupPrompt,
  buildVerifyMergePrompt,
} from "@/lib/prompts";

import type { WireImage } from "@/lib/images";

/**
 * OpenAI client.
 *
 * The API key should be stored in:
 * OPENAI_API_KEY
 *
 * Do NOT expose this key to browser/client-side code.
 */
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * OpenAI models.
 *
 * These can be overridden in Vercel Environment Variables:
 *
 * OPENAI_MODEL
 * OPENAI_CHECK_MODEL
 */
const GROUP_MODEL =
  process.env.OPENAI_MODEL || "gpt-5.5";

const CHECK_MODEL =
  process.env.OPENAI_CHECK_MODEL || GROUP_MODEL;

const BATCH_SIZE = 10;

/**
 * Concurrency caps.
 *
 * Keep these modest so a large sort does not create
 * a burst of simultaneous OpenAI requests.
 */
const GROUP_CONCURRENCY = 2;
const VERIFY_CONCURRENCY = 3;
const MERGE_CONCURRENCY = 2;

/**
 * The sort route runs under a Vercel max duration.
 *
 * Leave enough headroom for the route to return normally.
 */
export const SORT_TIME_BUDGET_MS = 250_000;

/**
 * Each individual OpenAI request gets its own budget.
 */
const PER_CALL_TIMEOUT_MS = 60_000;

/**
 * Don't start a request if there is less than this much
 * time remaining in the overall route budget.
 */
const MIN_CALL_MS = 5_000;

/**
 * OpenAI retryable HTTP statuses.
 *
 * The OpenAI SDK already performs some retries itself.
 * This loop is an additional application-level retry layer
 * for the specific long-running sort operation.
 */
const RETRYABLE_STATUS = new Set([
  408,
  409,
  429,
  500,
  502,
  503,
  504,
]);

export interface SortGroup {
  name: string;
  photoIndices: number[];
}

export interface SortResult {
  groups: SortGroup[];
  orphanIndices: number[];
}

/**
 * Thrown when an entire batch becomes unavailable because
 * OpenAI requests failed/rate-limited.
 */
export class SortUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SortUnavailableError";
  }
}

/**
 * Sleep helper.
 */
const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Extract text from an OpenAI Responses API response.
 *
 * The OpenAI SDK exposes response.output_text specifically
 * for this purpose.
 */
function getResponseText(response: OpenAI.Responses.Response): string {
  return response.output_text?.trim() ?? "";
}

/**
 * Normalize a group name so the same group doesn't get
 * returned under tiny formatting variations.
 */
function slugifyFolderName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/\s+/g, "-")
    .slice(0, 100);
}

/**
 * Run an async function over items with a fixed concurrency cap.
 *
 * Results stay in the same order as the original items.
 */
async function mapLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (items.length === 0) return [];

  const results = new Array<R>(items.length);
  let cursor = 0;

  const workers = Array.from(
    {
      length: Math.min(limit, items.length),
    },
    async () => {
      while (true) {
        const index = cursor++;

        if (index >= items.length) {
          return;
        }

        results[index] = await fn(items[index], index);
      }
    },
  );

  await Promise.all(workers);

  return results;
}

/**
 * Convert our application's WireImage into an OpenAI
 * Responses API input_image block.
 *
 * OpenAI accepts either:
 *
 * - a fully qualified image URL
 * - a base64 data URL
 *
 * We intentionally do NOT use Anthropic image blocks here.
 */
type OpenAIInputContent =
  | {
      type: "input_text";
      text: string;
    }
  | {
      type: "input_image";
      image_url: string;
      detail?: "low" | "high" | "auto";
    };

function toOpenAIImageInput(
  image: WireImage,
): OpenAIInputContent {
  /**
   * We don't assume the exact internal WireImage shape here.
   *
   * The application may store uploaded images as a URL,
   * imageUrl, dataUrl, src, data, or base64.
   */
  const raw = image as unknown as Record<string, unknown>;

  const possibleUrl =
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

  if (!possibleUrl) {
    throw new Error(
      "WireImage does not contain a usable image URL or data URL.",
    );
  }

  let imageUrl = possibleUrl;

  /**
   * If the stored value is raw base64 rather than a data URL,
   * turn it into an OpenAI-compatible data URL.
   */
  if (
    !imageUrl.startsWith("http://") &&
    !imageUrl.startsWith("https://") &&
    !imageUrl.startsWith("data:")
  ) {
    const mimeType =
      typeof raw.mimeType === "string"
        ? raw.mimeType
        : typeof raw.contentType === "string"
          ? raw.contentType
          : "image/jpeg";

    imageUrl = `data:${mimeType};base64,${imageUrl}`;
  }

  return {
    type: "input_image",
    image_url: imageUrl,
    detail: "auto",
  };
}

/**
 * Convert the model's JSON response into an object.
 *
 * OpenAI's Responses API returns text through response.output_text.
 *
 * We still support fenced JSON in case the model returns:
 *
 * ```json
 * {...}
 * ```
 */
export function parseModelJson<T = unknown>(
  raw: string,
): T {
  let text = raw.trim();

  if (!text) {
    throw new Error("OpenAI returned an empty response.");
  }

  text = text
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    return JSON.parse(text) as T;
  } catch {
    const match = text.match(/\{[\s\S]*\}/);

    if (match) {
      try {
        return JSON.parse(match[0]) as T;
      } catch {
        // Fall through.
      }
    }
  }

  throw new Error(
    `OpenAI did not return valid JSON. Response started with: ${text.slice(
      0,
      300,
    )}`,
  );
}

/**
 * Get a useful HTTP status from an OpenAI error.
 */
function getOpenAIErrorStatus(
  error: unknown,
): number | undefined {
  if (error instanceof OpenAI.APIError) {
    return error.status;
  }

  if (
    typeof error === "object" &&
    error !== null &&
    "status" in error
  ) {
    const status = (error as { status?: unknown }).status;

    return typeof status === "number" ? status : undefined;
  }

  return undefined;
}

/**
 * Get a useful error message.
 */
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  if (
    typeof error === "object" &&
    error !== null &&
    "message" in error
  ) {
    const message = (error as { message?: unknown }).message;

    if (typeof message === "string") {
      return message;
    }
  }

  return String(error);
}

/**
 * Main OpenAI JSON helper.
 *
 * This replaces the old Claude/Anthropic claudeJson() function.
 *
 * IMPORTANT:
 * This uses:
 *
 *     openai.responses.create()
 *
 * NOT:
 *
 *     client.messages.create()
 *
 * which was the Anthropic API.
 */
async function openAIJson<T = unknown>(params: {
  model: string;
  content: OpenAIInputContent[];
  maxOutputTokens: number;
  label: string;
  deadline: number;
}): Promise<T | null> {
  const {
    model,
    content,
    maxOutputTokens,
    label,
    deadline,
  } = params;

  for (let attempt = 0; attempt < 4; attempt++) {
    const remaining = deadline - Date.now();

    if (remaining < MIN_CALL_MS) {
      console.warn(
        `[sort] ${label}: time budget exhausted - skipping call`,
      );

      return null;
    }

    const timeout = Math.min(
      PER_CALL_TIMEOUT_MS,
      Math.max(MIN_CALL_MS, remaining - 1_000),
    );

    try {
      const response = await openai.responses.create(
        {
          model,
          input: [
            {
              role: "user",
              content,
            },
          ],

          /**
           * Ask OpenAI to return JSON.
           *
           * The existing prompts already explain the desired
           * object shape, so we keep parsing flexible here.
           */
          text: {
            format: {
              type: "json_object",
            },
          },

          max_output_tokens: maxOutputTokens,

          /**
           * We don't need server-side conversation state for
           * these independent sorting calls.
           */
          store: false,
        },
        {
          timeout,
          maxRetries: 0,
        },
      );

      const text = getResponseText(response);

      return parseModelJson<T>(text);
    } catch (error) {
      const status = getOpenAIErrorStatus(error);

      const retryable =
        status !== undefined &&
        RETRYABLE_STATUS.has(status);

      if (retryable && attempt < 3) {
        const remaining = deadline - Date.now();

        if (remaining < MIN_CALL_MS) {
          console.warn(
            `[sort] ${label}: no time left for retry`,
          );

          return null;
        }

        const waitMs = Math.min(
          8_000,
          1_000 * 2 ** attempt +
            Math.floor(Math.random() * 500),
        );

        console.warn(
          `[sort] ${label}: OpenAI ${status}; retry ${
            attempt + 1
          } in ${waitMs}ms`,
        );

        await sleep(waitMs);
        continue;
      }

      console.warn(
        `[sort] ${label}: OpenAI request failed`,
        {
          status,
          error: getErrorMessage(error),
        },
      );

      return null;
    }
  }

  return null;
}

/**
 * Step 1:
 * Group photos in independent batches.
 *
 * The important difference from the old implementation is
 * that every image is now sent as an OpenAI input_image.
 */
async function groupPhotos(params: {
  images: WireImage[];
  model: string;
  deadline: number;
  name?: string;
  indices: number[];
}): Promise<
  Array<{
    name: string;
    indices: number[];
  }> | null
> {
  const {
    images,
    model,
    deadline,
    name,
    indices,
  } = params;

  const total = images.length;

  const batches: Array<{
    offset: number;
    batch: WireImage[];
    labelStart: number;
    labelEnd: number;
  }> = [];

  for (
    let offset = 0;
    offset < total;
    offset += BATCH_SIZE
  ) {
    const batch = images.slice(
      offset,
      offset + BATCH_SIZE,
    );

    batches.push({
      offset,
      batch,
      labelStart: offset + 1,
      labelEnd: offset + batch.length,
    });
  }

  const perBatch = await mapLimit(
    batches,
    GROUP_CONCURRENCY,
    async (batch) => {
      const remaining = deadline - Date.now();

      if (remaining < MIN_CALL_MS) {
        return {
          out: null,
          failed: true,
        };
      }

      const note =
        batch.offset > 0
          ? `These are photos ${batch.labelStart}-${batch.labelEnd} of ${total} total. Group only the photos shown in this request.`
          : "";

      const prompt = buildSortPrompt(
        batch.batch.length,
        batch.labelStart,
        batch.labelEnd,
        note,
      );

      const content: OpenAIInputContent[] = [
        {
          type: "input_text",
          text: prompt,
        },
      ];

      for (const image of batch.batch) {
        content.push(toOpenAIImageInput(image));
      }

      const data = await openAIJson<{
        groups?: Array<{
          name?: string;
          photo_indices?: number[];
          indices?: number[];
        }>;
      }>({
        model,
        content,
        maxOutputTokens: 2_000,
        label: `${name ?? "sort"} batch ${batch.labelStart}-${batch.labelEnd}`,
        deadline,
      });

      if (!data) {
        return {
          out: null,
          failed: true,
        };
      }

      const groups: Array<{
        name: string;
        indices: number[];
      }> = [];

      for (const group of data.groups ?? []) {
        const rawName =
          typeof group.name === "string"
            ? group.name
            : "";

        const name = rawName.trim();

        if (!name) continue;

        const sourceIndices =
          Array.isArray(group.photo_indices)
            ? group.photo_indices
            : Array.isArray(group.indices)
              ? group.indices
              : [];

        const validIndices = sourceIndices
          .filter(
            (index): index is number =>
              Number.isInteger(index),
          )
          .filter(
            (index) =>
              index >= batch.labelStart &&
              index <= batch.labelEnd,
          );

        groups.push({
          name,
          indices: validIndices,
        });
      }

      return {
        out: groups,
        failed: false,
      };
    },
  );

  if (
    perBatch.length > 0 &&
    perBatch.every((batch) => batch.failed)
  ) {
    throw new SortUnavailableError(
      "The photo-sorting service was unavailable or rate-limited. Wait a minute and try again; reducing the number of photos won't help if every request is failing.",
    );
  }

  const output: Array<{
    name: string;
    indices: number[];
  }> = [];

  for (const batch of perBatch) {
    if (!batch.out) continue;

    output.push(...batch.out);
  }

  return output;
}

/**
 * Step 2:
 * Verify groups that contain multiple photos.
 *
 * This catches accidental mixed groups.
 */
async function verifyGroups(params: {
  client?: unknown;
  images: WireImage[];
  groups: Array<{
    name: string;
    indices: number[];
  }>;
  model: string;
  deadline: number;
}): Promise<{
  groups: Array<{
    name: string;
    indices: number[];
  }>;
  orphans: number[];
}> {
  const {
    images,
    groups,
    model,
    deadline,
  } = params;

  const orphans: number[] = [];

  const verified = await mapLimit(
    groups,
    VERIFY_CONCURRENCY,
    async (group) => {
      if (group.indices.length <= 1) {
        return group;
      }

      const imageIndices = group.indices
        .slice()
        .sort((a, b) => a - b);

      const firstIndex = imageIndices[0];

      if (
        firstIndex === undefined ||
        !images[firstIndex - 1]
      ) {
        return group;
      }

      const content: OpenAIInputContent[] = [
        {
          type: "input_text",
          text: buildVerifyGroupPrompt(
            group.name,
            group.indices.length,
          ),
        },
      ];

      for (const index of imageIndices) {
        const image = images[index - 1];

        if (!image) continue;

        content.push({
          type: "input_text",
          text: `Photo ${index}:`,
        });

        content.push(
          toOpenAIImageInput(image),
        );
      }

      const result = await openAIJson<{
        valid?: boolean;
        keep?: number[];
        remove?: number[];
      }>({
        model,
        content,
        maxOutputTokens: 1_000,
        label: `verify ${group.name}`,
        deadline,
      });

      if (!result) {
        /**
         * Fail safe:
         * if verification is unavailable, preserve the
         * original grouping rather than throwing away photos.
         */
        return group;
      }

      if (result.valid === false) {
        const remove = new Set(
          Array.isArray(result.remove)
            ? result.remove.filter(
                (n): n is number =>
                  Number.isInteger(n),
              )
            : [],
        );

        const kept = group.indices.filter(
          (index) => !remove.has(index),
        );

        for (const index of remove) {
          if (group.indices.includes(index)) {
            orphans.push(index);
          }
        }

        if (kept.length === 0) {
          return null;
        }

        return {
          name: group.name,
          indices: kept,
        };
      }

      if (Array.isArray(result.keep)) {
        const allowed = new Set(
          result.keep.filter(
            (n): n is number =>
              Number.isInteger(n),
          ),
        );

        const kept = group.indices.filter(
          (index) => allowed.has(index),
        );

        for (const index of group.indices) {
          if (!allowed.has(index)) {
            orphans.push(index);
          }
        }

        if (kept.length === 0) {
          return null;
        }

        return {
          name: group.name,
          indices: kept,
        };
      }

      return group;
    },
  );

  return {
    groups: verified.filter(
      (
        group,
      ): group is {
        name: string;
        indices: number[];
      } => group !== null,
    ),
    orphans,
  };
}

/**
 * Step 3:
 * Merge adjacent groups that are really one split group.
 *
 * Only neighboring groups are compared, keeping this relatively
 * inexpensive while still catching chunk-boundary splits.
 */
async function mergeAdjacentGroups(params: {
  images: WireImage[];
  groups: Array<{
    name: string;
    indices: number[];
  }>;
  model: string;
  deadline: number;
}): Promise<
  Array<{
    name: string;
    indices: number[];
  }>
> {
  const {
    images,
    groups,
    model,
    deadline,
  } = params;

  if (groups.length < 2) {
    return groups;
  }

  const merged: Array<{
    name: string;
    indices: number[];
  }> = [];

  let i = 0;

  while (i < groups.length) {
    const current = groups[i];
    const next = groups[i + 1];

    if (!current) {
      i += 1;
      continue;
    }

    if (!next) {
      merged.push(current);
      i += 1;
      continue;
    }

    const currentLast =
      current.indices[current.indices.length - 1];

    const nextFirst = next.indices[0];

    if (
      currentLast === undefined ||
      nextFirst === undefined ||
      !images[currentLast - 1] ||
      !images[nextFirst - 1]
    ) {
      merged.push(current);
      i += 1;
      continue;
    }

    const content: OpenAIInputContent[] = [
      {
        type: "input_text",
        text: "Group A:",
      },
      {
        type: "input_text",
        text: `Existing group: ${current.name}`,
      },
      toOpenAIImageInput(images[currentLast - 1]),

      {
        type: "input_text",
        text: "Group B:",
      },
      {
        type: "input_text",
        text: `Existing group: ${next.name}`,
      },
      toOpenAIImageInput(images[nextFirst - 1]),

      {
        type: "input_text",
        text: buildVerifyMergePrompt(
          current.indices.length,
          next.indices.length,
        ),
      },
    ];

    const result = await openAIJson<{
      merge?: boolean;
      merged_name?: string;
      name?: string;
    }>({
      model,
      content,
      maxOutputTokens: 500,
      label: `merge ${current.name} + ${next.name}`,
      deadline,
    });

    if (result?.merge === true) {
      const mergedName =
        typeof result.merged_name === "string"
          ? result.merged_name
          : typeof result.name === "string"
            ? result.name
            : current.name;

      merged.push({
        name: mergedName,
        indices: [
          ...current.indices,
          ...next.indices,
        ].sort((a, b) => a - b),
      });

      i += 2;
      continue;
    }

    merged.push(current);
    i += 1;
  }

  return merged;
}

/**
 * Combine duplicate names after the merge step.
 */
function uniqueNames(
  groups: Array<{
    name: string;
    indices: number[];
  }>,
): SortGroup[] {
  const byName = new Map<
    string,
    {
      name: string;
      indices: number[];
    }
  >();

  for (const group of groups) {
    const cleanName = group.name.trim();

    if (!cleanName) continue;

    const key = slugifyFolderName(cleanName);

    const existing = byName.get(key);

    if (existing) {
      existing.indices.push(...group.indices);
    } else {
      byName.set(key, {
        name: cleanName,
        indices: [...group.indices],
      });
    }
  }

  return Array.from(byName.values()).map(
    (group) => ({
      name: group.name,
      photoIndices: Array.from(
        new Set(group.indices),
      ).sort((a, b) => a - b),
    }),
  );
}

/**
 * Public sorting function.
 *
 * This is the main entry point used by the route.
 */
export async function sortPhotos(params: {
  client?: unknown;
  images: WireImage[];
  model?: string;
  budgetMs?: number;
}): Promise<SortResult> {
  const {
    images,
    model = GROUP_MODEL,
    budgetMs = SORT_TIME_BUDGET_MS,
  } = params;

  if (images.length === 0) {
    return {
      groups: [],
      orphanIndices: [],
    };
  }

  const deadline = Date.now() + budgetMs;

  console.log(
    `[sort] starting OpenAI photo sort: ${images.length} photos using ${model}`,
  );

  /**
   * Step 1:
   * Sort photos in batches.
   */
  const initialGroups = await groupPhotos({
    images,
    model,
    deadline,
    indices: images.map((_, index) => index + 1),
  });

  if (!initialGroups) {
    throw new SortUnavailableError(
      "OpenAI sorting did not return a result.",
    );
  }

  /**
   * Step 2:
   * Verify each multi-photo group.
   */
  const verified = await verifyGroups({
    images,
    groups: initialGroups,
    model: CHECK_MODEL,
    deadline,
  });

  /**
   * Step 3:
   * Merge adjacent groups when OpenAI determines they are
   * actually one item/group split across a batch boundary.
   */
  const merged = await mergeAdjacentGroups({
    images,
    groups: verified.groups,
    model: CHECK_MODEL,
    deadline,
  });

  /**
   * Determine which photos were never assigned to a group.
   */
  const assigned = new Set<number>();

  for (const group of merged) {
    for (const index of group.indices) {
      assigned.add(index);
    }
  }

  const orphanIndices = Array.from(
    new Set([
      ...verified.orphans,
      ...images
        .map((_, index) => index + 1)
        .filter((index) => !assigned.has(index)),
    ]),
  ).sort((a, b) => a - b);

  return {
    groups: uniqueNames(merged),
    orphanIndices,
  };
}
