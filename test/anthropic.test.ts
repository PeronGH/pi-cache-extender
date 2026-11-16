import { describe, expect, test } from "bun:test";
import type { AssistantMessage, Model, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { buildWarmPayload, isReplayableAnthropicPayload, sendWarmPing } from "../src/anthropic.ts";

const haiku: Model<"anthropic-messages"> = {
	id: "claude-haiku-5-5",
	name: "Claude Haiku 5.5",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.githubcopilot.com",
	reasoning: true,
	input: ["text"],
	cost: {
		input: 0.1,
		output: 0.5,
		cacheRead: 0.01,
		cacheWrite: 0.125,
		tiers: [{ inputTokensAbove: 100000, input: 0.5, output: 2.5, cacheRead: 0.05, cacheWrite: 0.625 }],
	},
	contextWindow: 1000000,
	maxTokens: 128000,
	promptCache: { short: 300, long: 3600 },
};

describe("anthropic payload validation and streamSimple warm ping", () => {
	test("accepts adaptive thinking payload with cache_control and rejects budget thinking or uncached", () => {
		const adaptive = {
			model: "claude-haiku-5-5",
			thinking: { type: "adaptive", display: "summarized" },
			system: [{ type: "text", text: "sys", cache_control: { type: "ephemeral" } }],
			messages: [{ role: "user", content: "hi" }],
		};
		expect(isReplayableAnthropicPayload(adaptive, haiku)).toBe(true);

		const budget = { ...adaptive, thinking: { type: "enabled", budget_tokens: 1024 } };
		expect(isReplayableAnthropicPayload(budget, haiku)).toBe(false);

		const noPromptCache = { ...haiku, promptCache: undefined };
		expect(isReplayableAnthropicPayload(adaptive, noPromptCache)).toBe(false);

		const uncached = {
			model: "claude-haiku-5-5",
			system: [{ type: "text", text: "sys" }],
			messages: [{ role: "user", content: "hi" }],
		};
		expect(isReplayableAnthropicPayload(uncached, haiku)).toBe(false);
	});

	test("buildWarmPayload strips thread and sets max_tokens=0 and stream=true", () => {
		const built = buildWarmPayload(
			{
				model: "claude-haiku-5-5",
				max_tokens: 4096,
				stream: false,
				betas: ["mid-conversation-output-config-2026-07-01"],
				thread: { type: "create" },
				thinking: { type: "adaptive" },
				messages: [{ role: "user", content: [{ type: "text", text: "hi", cache_control: { type: "ephemeral" } }] }],
			},
			0,
		);
		expect(built.max_tokens).toBe(0);
		expect(built.stream).toBe(true);
		expect(built.thread).toBeUndefined();
		expect(built.betas).toEqual(["mid-conversation-output-config-2026-07-01"]);
	});

	test("sendWarmPing delegates to modelRegistry.streamSimple with onPayload and detects max_tokens rejection", async () => {
		let capturedOptions: SimpleStreamOptions | undefined;
		const makeFakeRegistry = (reply: Partial<AssistantMessage>) => ({
			streamSimple: (_model: Model<any>, _ctx: any, options?: SimpleStreamOptions) => {
				capturedOptions = options;
				return {
					result: async () =>
						({
							role: "assistant",
							content: [],
							api: "anthropic-messages",
							provider: "anthropic",
							model: "claude-haiku-5-5",
							usage: {
								input: 4,
								output: 0,
								cacheRead: 2985,
								cacheWrite: 0,
								totalTokens: 2989,
								cost: { input: 0.0000004, output: 0, cacheRead: 0.00002985, cacheWrite: 0, total: 0.00003025 },
							},
							stopReason: "length",
							timestamp: Date.now(),
							...reply,
						}) as AssistantMessage,
				} as any;
			},
		});

		const snapshot = {
			model: haiku,
			capturedAt: 1000,
			payload: {
				model: "claude-haiku-5-5",
				max_tokens: 4096,
				messages: [{ role: "user", content: "hi" }],
			},
		};

		const okRes = await sendWarmPing(makeFakeRegistry({}), snapshot, new AbortController().signal, 0);
		expect(okRes.ok).toBe(true);
		expect(okRes.usage?.cacheRead).toBe(2985);
		expect(capturedOptions?.disableContinuation).toBe(true);
		expect(capturedOptions?.maxRetries).toBe(0);
		const replaced = await capturedOptions?.onPayload?.({ dummy: true }, haiku);
		expect((replaced as Record<string, unknown>).max_tokens).toBe(0);

		const errRes = await sendWarmPing(
			makeFakeRegistry({ stopReason: "error", errorMessage: "max_tokens must be >= 1" }),
			snapshot,
			new AbortController().signal,
			0,
		);
		expect(errRes.ok).toBe(false);
		expect(errRes.rejectedZeroMaxTokens).toBe(true);
	});
});
