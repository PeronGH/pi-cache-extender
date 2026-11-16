import { afterEach, beforeEach, expect, jest, test } from "bun:test";
import type { Model } from "@earendil-works/pi-ai";
import extension from "../extensions/index.ts";

const MINUTE = 60_000;

const model: Model<"anthropic-messages"> = {
	id: "claude-fable-5-2",
	name: "Claude Fable 5.2",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: true,
	input: ["text"],
	cost: { input: 8, output: 40, cacheRead: 0.2, cacheWrite: 10 },
	contextWindow: 1000000,
	maxTokens: 128000,
	promptCache: { short: 300, long: 3600 },
};

const usage = {
	input: 10,
	output: 1,
	cacheRead: 100_000,
	cacheWrite: 0,
	totalTokens: 100_011,
	cost: { input: 0, output: 0, cacheRead: 0.02, cacheWrite: 0, total: 0.02 },
};

function payload(ttl?: "1h") {
	return {
		model: model.id,
		max_tokens: 4096,
		system: [{ type: "text", text: "sys", cache_control: { type: "ephemeral", ...(ttl && { ttl }) } }],
		messages: [{ role: "user", content: "hi" }],
	};
}

/** Load the extension against a fake pi that records the refreshes it sends. */
function load() {
	const handlers = new Map<string, (event: unknown, ctx: unknown) => unknown>();
	let command: ((args: string, ctx: unknown) => Promise<void>) | undefined;
	const pings: unknown[] = [];
	const notes: string[] = [];
	const ctx = {
		model,
		modelRegistry: {
			streamSimple: (_model: unknown, _context: unknown, options: { onPayload: () => unknown }) => ({
				result: async () => {
					pings.push(options.onPayload());
					return { role: "assistant", stopReason: "length", usage };
				},
			}),
		},
		sessionManager: { getBranch: () => [{ type: "message", message: { role: "assistant", usage } }] },
		ui: { notify: (text: string) => notes.push(text) },
	};
	extension({
		registerFlag() {},
		getFlag: () => undefined,
		appendEntry() {},
		on: (name: string, handler: (event: unknown, ctx: unknown) => unknown) => handlers.set(name, handler),
		registerCommand: (_name: string, spec: { handler: typeof command }) => {
			command = spec.handler;
		},
	} as any);
	return {
		pings,
		emit: (name: string, event: unknown = {}) => handlers.get(name)?.(event, ctx),
		status: async (args = "") => {
			await command?.(args, ctx);
			return notes.at(-1) ?? "";
		},
	};
}

/** Advance the fake clock in small steps so each refresh settles before the next one comes due. */
async function advance(ms: number) {
	for (let elapsed = 0; elapsed < ms; elapsed += 10_000) {
		jest.advanceTimersByTime(Math.min(10_000, ms - elapsed));
		for (let turn = 0; turn < 20; turn++) await Promise.resolve();
	}
}

beforeEach(() => {
	jest.useFakeTimers({ now: 1_000_000_000 });
});

afterEach(() => {
	jest.useRealTimers();
});

test("keeps an idle cache warm past 30 minutes, until the curve says stop", async () => {
	const pi = load();
	pi.emit("before_provider_request", { payload: payload() });
	pi.emit("agent_settled");

	await advance(35 * MINUTE);
	expect(pi.pings.length).toBe(7);
	expect(pi.pings[0]).toEqual({ ...payload(), max_tokens: 1 });

	await advance(4 * 60 * MINUTE);
	expect(pi.pings.length).toBeLessThan(20);
	expect(await pi.status()).toContain("stopped (past curve horizon)");
});

test("stops refreshing a run that stays active for over an hour", async () => {
	const pi = load();
	pi.emit("before_provider_request", { payload: payload() });

	await advance(3 * 60 * MINUTE);

	expect(pi.pings.length).toBe(13);
	expect(await pi.status()).toContain("stopped (run active for over an hour)");
});

test("changing the curve keeps the pending refresh on schedule", async () => {
	const pi = load();
	pi.emit("before_provider_request", { payload: payload() });
	pi.emit("agent_settled");

	await advance(3 * MINUTE);
	await pi.status("15m 90m");
	await advance(1.5 * MINUTE);

	expect(pi.pings.length).toBe(1);
});

test("changing the curve after the cache expired does not send a refresh", async () => {
	const pi = load();
	await pi.status("10s 20s");
	pi.emit("before_provider_request", { payload: payload() });
	pi.emit("agent_settled");
	await advance(10 * MINUTE);
	expect(await pi.status()).toContain("stopped (past curve horizon)");

	expect(await pi.status("60m 120m")).toContain("stopped (cache expired)");
	await advance(10 * MINUTE);

	expect(pi.pings.length).toBe(0);
});

test("refreshes a one-hour cache entry once per 54 minutes", async () => {
	const pi = load();
	await pi.status("3h 10h");
	pi.emit("before_provider_request", { payload: payload("1h") });
	pi.emit("agent_settled");

	await advance(53 * MINUTE);
	expect(pi.pings.length).toBe(0);

	await advance(2 * MINUTE);
	expect(pi.pings.length).toBe(1);
});

test("takes over from pi's warmer only for requests it can replay", () => {
	const pi = load();
	pi.emit("before_provider_request", { payload: payload() });
	expect(pi.emit("cache_warming_decision")).toEqual({ action: "stop" });

	pi.emit("before_provider_request", { payload: { ...payload(), thinking: { type: "enabled", budget_tokens: 1024 } } });
	expect(pi.emit("cache_warming_decision")).toBeUndefined();
});
