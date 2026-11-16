import { describe, expect, test } from "bun:test";
import type { Model, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { captureRequest, sendWarmPing } from "../src/anthropic.ts";

const haiku: Model<"anthropic-messages"> = {
	id: "claude-haiku-5-5",
	name: "Claude Haiku 5.5",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: true,
	input: ["text"],
	cost: { input: 0.1, output: 0.5, cacheRead: 0.01, cacheWrite: 0.125 },
	contextWindow: 1000000,
	maxTokens: 128000,
	promptCache: { short: 300, long: 3600 },
};

const payload = {
	model: "claude-haiku-5-5",
	max_tokens: 4096,
	thinking: { type: "adaptive" },
	system: [{ type: "text", text: "sys", cache_control: { type: "ephemeral" } }],
	messages: [{ role: "user", content: "hi" }],
};

describe("captureRequest", () => {
	test("captures a cached request with the lifetime of its retention tier", () => {
		expect(captureRequest(payload, haiku)).toMatchObject({ ttlMs: 300_000, delayMs: 270_000, payload });

		const long = {
			...payload,
			system: [{ type: "text", text: "sys", cache_control: { type: "ephemeral", ttl: "1h" } }],
		};
		expect(captureRequest(long, haiku)).toMatchObject({ ttlMs: 3_600_000, delayMs: 3_240_000 });
	});

	test("rejects requests a one-token replay cannot refresh", () => {
		expect(captureRequest({ ...payload, thinking: { type: "enabled", budget_tokens: 1024 } }, haiku)).toBeUndefined();
		expect(captureRequest({ ...payload, system: [{ type: "text", text: "sys" }] }, haiku)).toBeUndefined();
		expect(captureRequest({ ...payload, model: "claude-opus-5-5" }, haiku)).toBeUndefined();
		expect(captureRequest(payload, { ...haiku, promptCache: undefined })).toBeUndefined();
		expect(captureRequest(payload, { ...haiku, api: "openai-responses" })).toBeUndefined();
	});
});

test("sendWarmPing replays the captured payload capped at one output token", async () => {
	let options: SimpleStreamOptions | undefined;
	const registry = {
		streamSimple: (_model: unknown, _context: unknown, streamOptions?: SimpleStreamOptions) => {
			options = streamOptions;
			return { result: async () => ({ stopReason: "length" }) } as any;
		},
	};
	const request = captureRequest(payload, haiku);
	if (!request) throw new Error("expected a replayable request");

	await sendWarmPing(registry, request, new AbortController().signal);

	expect(options?.maxRetries).toBe(0);
	expect(await options?.onPayload?.({}, haiku)).toEqual({ ...payload, max_tokens: 1 });
});
