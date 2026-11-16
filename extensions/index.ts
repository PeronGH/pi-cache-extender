import { CACHE_WARMING_MAX_AGE_MS, evaluateCacheWarming } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type CapturedRequest, captureRequest, sendWarmPing } from "../src/anthropic.ts";
import {
	type CacheExtenderConfig,
	DEFAULT_CONFIG,
	formatCurveSummary,
	formatDuration,
	parseCommandArgs,
	plannedRefreshes,
} from "../src/policy.ts";

const ENTRY_TYPE = "cache-extender-config";

type ConfigSource = "default" | "flag" | "session";

function isValidConfig(value: unknown): value is CacheExtenderConfig {
	if (!value || typeof value !== "object") return false;
	const record = value as Record<string, unknown>;
	if (record.mode === "off") return true;
	return (
		record.mode === "curve" &&
		typeof record.t90Ms === "number" &&
		typeof record.t10Ms === "number" &&
		typeof record.t50Ms === "number" &&
		typeof record.k === "number" &&
		record.t90Ms > 0 &&
		record.t10Ms > record.t90Ms &&
		record.t50Ms > 0 &&
		record.k > 0
	);
}

function formatCost(usd: number): string {
	if (usd <= 0) return "$0";
	if (usd < 0.001) return `$${usd.toFixed(5)}`;
	if (usd < 0.01) return `$${usd.toFixed(4)}`;
	return `$${usd.toFixed(3)}`;
}

/** Prompt size of the most recent request on the branch that reported usage. */
function lastPromptTokens(ctx: ExtensionContext): number {
	const branch = ctx.sessionManager.getBranch();
	for (let index = branch.length - 1; index >= 0; index--) {
		const entry = branch[index];
		if (entry.type !== "message" || entry.message.role !== "assistant") continue;
		const { input, cacheRead, cacheWrite } = entry.message.usage;
		if (input + cacheRead + cacheWrite > 0) return input + cacheRead + cacheWrite;
	}
	return 0;
}

export default function (pi: ExtensionAPI) {
	let config: CacheExtenderConfig = DEFAULT_CONFIG;
	let configSource: ConfigSource = "default";
	/** The latest agent request, while this extension owns warming its cache entry. */
	let request: CapturedRequest | undefined;
	/** When `request` was sent. */
	let requestedAt = 0;
	/** Start of the last request that read or wrote the cache entry, which is when its lifetime restarted. */
	let touchedAt = 0;
	/** Undefined while an agent run is active. */
	let idleSince: number | undefined;
	let nextWarmAt = 0;
	let warmTimer: ReturnType<typeof setTimeout> | undefined;
	let inFlight: AbortController | undefined;
	let stopReason: string | undefined;
	let warmCount = 0;
	let totalWarmCost = 0;

	pi.registerFlag("cache-ttl", {
		description: "Cache return curve (e.g. '12m 60m', '30m', 'off')",
		type: "string",
	});

	const stop = (reason?: string) => {
		clearTimeout(warmTimer);
		warmTimer = undefined;
		inFlight?.abort();
		inFlight = undefined;
		stopReason = reason;
	};

	const forget = (reason?: string) => {
		request = undefined;
		stop(reason);
	};

	/** Number of refreshes worth sending from time `at` on, or the reason to send none. */
	const plan = (ctx: ExtensionContext, target: CapturedRequest, at: number): number | string => {
		if (config.mode === "off") return "off";
		if (idleSince === undefined) {
			// The agent will send its next request for sure, unless a tool hangs: bound that like pi does.
			const remainingMs = requestedAt + CACHE_WARMING_MAX_AGE_MS - at;
			return remainingMs >= 0 ? Math.floor(remainingMs / target.delayMs) + 1 : "run active for over an hour";
		}
		const economics = evaluateCacheWarming(target.model, lastPromptTokens(ctx), "idle");
		if (!economics.economicsAvailable) return "cache economics unavailable";
		return (
			plannedRefreshes(config, at - idleSince, target.delayMs, economics.warmCost, economics.missCost) ||
			"past curve horizon"
		);
	};

	/** Arm the refresh that is due before the cache entry of `request` expires. */
	const schedule = (ctx: ExtensionContext, target: CapturedRequest) => {
		stop();
		nextWarmAt = touchedAt + target.delayMs;
		// A timer can run late after sleep. Past half of the pre-expiry margin the entry has likely expired, and the
		// refresh would be a full-price cache write.
		const deadline = nextWarmAt + Math.floor((target.ttlMs - target.delayMs) / 2);
		if (Date.now() > deadline) {
			stop("cache expired");
			return;
		}
		warmTimer = setTimeout(() => void warm(ctx, target, deadline), Math.max(0, nextWarmAt - Date.now()));
		warmTimer.unref?.();
	};

	// Every change of `request` clears the timer and aborts the refresh in flight, so `target` is current here.
	const warm = async (ctx: ExtensionContext, target: CapturedRequest, deadline: number) => {
		warmTimer = undefined;
		const startedAt = Date.now();
		const planned = startedAt > deadline ? "cache expired" : plan(ctx, target, startedAt);
		if (typeof planned === "string") {
			stop(planned);
			return;
		}

		const controller = new AbortController();
		inFlight = controller;
		// A rejection here would be unhandled and take pi down with it.
		const message = await sendWarmPing(ctx.modelRegistry, target, controller.signal).catch(() => undefined);
		if (inFlight !== controller) return;
		inFlight = undefined;

		if (!message || message.stopReason === "error" || message.stopReason === "aborted") {
			stop(message?.errorMessage ?? "refresh failed");
			return;
		}
		warmCount++;
		totalWarmCost += message.usage.cost.total;
		if (message.usage.cacheRead === 0) {
			stop("cache expired");
			return;
		}
		touchedAt = startedAt;
		schedule(ctx, target);
	};

	const restoreConfig = (ctx: ExtensionContext) => {
		config = DEFAULT_CONFIG;
		configSource = "default";
		const flagValue = pi.getFlag("cache-ttl");
		if (typeof flagValue === "string") {
			const parsedFlag = parseCommandArgs(flagValue);
			if (parsedFlag.action === "set") {
				config = parsedFlag.config;
				configSource = "flag";
			}
		}
		for (const entry of ctx.sessionManager.getBranch()) {
			if (entry.type === "custom" && entry.customType === ENTRY_TYPE && isValidConfig(entry.data)) {
				config = entry.data;
				configSource = "session";
			}
		}
	};

	const describeState = (ctx: ExtensionContext): string => {
		if (config.mode === "off") return "off, pi's built-in warmer applies";
		if (inFlight) return "refreshing now";
		if (!request || !warmTimer) return stopReason ? `stopped (${stopReason})` : "waiting for a request";
		const now = Date.now();
		const wait = formatDuration(Math.max(0, nextWarmAt - now));
		const planned = plan(ctx, request, nextWarmAt);
		if (typeof planned === "string") return `stopping in ${wait} (${planned})`;
		const warmUntil = nextWarmAt + (planned - 1) * request.delayMs + request.ttlMs;
		return `next refresh in ${wait}, up to ${planned} planned, keeping the cache for ${formatDuration(warmUntil - now)}`;
	};

	pi.on("session_start", (_event, ctx) => {
		restoreConfig(ctx);
	});

	pi.on("session_tree", (_event, ctx) => {
		forget();
		restoreConfig(ctx);
	});

	pi.on("session_shutdown", () => {
		stop();
	});

	pi.on("session_compact", () => {
		forget("compacted");
	});

	pi.on("model_select", (event) => {
		if (request && (event.model.provider !== request.model.provider || event.model.id !== request.model.id)) {
			forget("model changed");
		}
	});

	pi.on("before_provider_request", (event, ctx) => {
		idleSince = undefined;
		if (config.mode === "off") return;
		request = captureRequest(event.payload, ctx.model);
		if (!request) {
			stop("request cannot be replayed");
			return;
		}
		requestedAt = touchedAt = Date.now();
		schedule(ctx, request);
	});

	pi.on("agent_settled", () => {
		idleSince = Date.now();
	});

	pi.on("cache_warming_decision", () => {
		if (request) return { action: "stop" };
	});

	pi.registerCommand("cache-ttl", {
		description: "Configure cache return curve (e.g. /cache-ttl 12m 60m, 30m, off, reset)",
		getArgumentCompletions: (prefix) => {
			const presets = ["1.5m 3h", "15m 90m", "30m", "off", "reset"];
			const filtered = presets
				.filter((item) => item.startsWith(prefix.trim().toLowerCase()))
				.map((item) => ({ value: item, label: item }));
			return filtered.length > 0 ? filtered : null;
		},
		handler: async (args, ctx) => {
			const parsed = parseCommandArgs(args);
			if (parsed.action === "invalid") {
				ctx.ui.notify(parsed.message, "warning");
				return;
			}
			if (parsed.action === "set") {
				config = parsed.config;
				configSource = "session";
				pi.appendEntry(ENTRY_TYPE, config);
				// A pending refresh reads the new curve when it comes due. One that had stopped resumes only while
				// the cache entry is still alive.
				if (config.mode === "off") forget();
				else if (request && !warmTimer && !inFlight) schedule(ctx, request);
			}

			const lines = [
				config.mode === "off"
					? `Cache Extender: off (${configSource})`
					: `Cache Extender: ${formatCurveSummary(config)} (${configSource}) — 90% @ ${formatDuration(config.t90Ms)}, 50% @ ${formatDuration(config.t50Ms)}, 10% @ ${formatDuration(config.t10Ms)}, k=${config.k.toFixed(2)}`,
				`Status: ${describeState(ctx)}`,
			];
			if (request) {
				const tokens = lastPromptTokens(ctx);
				const economics = evaluateCacheWarming(request.model, tokens, "idle");
				const costs = economics.economicsAvailable
					? `, refresh ${formatCost(economics.warmCost)} vs miss ${formatCost(economics.missCost)}`
					: "";
				lines.push(
					`Cache: ${request.model.id}, ${tokens.toLocaleString()} tokens, ${formatDuration(request.ttlMs)} lifetime${costs}`,
				);
			}
			if (warmCount > 0) lines.push(`Session warmed: ${warmCount}x (${formatCost(totalWarmCost)})`);
			if (parsed.action === "status") lines.push("Usage: /cache-ttl [t90 t10 | t50 | off | reset]");
			ctx.ui.notify(lines.join("\n"), "info");
		},
	});
}
