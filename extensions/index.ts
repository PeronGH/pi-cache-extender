import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type CapturedAnthropicRequest, isReplayableAnthropicPayload, sendWarmPing } from "../src/anthropic.ts";
import {
	type CacheExtenderConfig,
	computeCacheEconomics,
	DEFAULT_CONFIG,
	evaluateOptimalStopping,
	formatCurveSummary,
	formatDuration,
	getRefreshSchedule,
	parseCommandArgs,
	type StoppingDecision,
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

export default function (pi: ExtensionAPI) {
	let config: CacheExtenderConfig = DEFAULT_CONFIG;
	let configSource: ConfigSource = "default";
	let snapshot: CapturedAnthropicRequest | undefined;
	let promptTokens = 0;
	let phase: "streaming" | "idle" = "idle";
	let inActiveTurn = false;
	let idleSince = 0;
	let nextWarmAt = 0;
	let refreshDeadlineAt = 0;
	let warmCount = 0;
	let totalWarmCost = 0;
	let lastDecision: StoppingDecision | undefined;
	let stopReason: string | undefined;
	let forceOneToken = false;
	let sessionEpoch = 0;
	let warmTimer: ReturnType<typeof setTimeout> | undefined;
	let inFlight: AbortController | undefined;

	pi.registerFlag("cache-ttl", {
		description: "Cache return curve (e.g. '12m 60m', '30m', 'off')",
		type: "string",
	});

	const clearTimers = () => {
		if (warmTimer) {
			clearTimeout(warmTimer);
			warmTimer = undefined;
		}
		if (inFlight) {
			inFlight.abort();
			inFlight = undefined;
		}
		nextWarmAt = 0;
		refreshDeadlineAt = 0;
	};

	const stopWarming = (reason?: string) => {
		clearTimers();
		lastDecision = undefined;
		stopReason = reason;
	};

	const getModelTtlMs = (req: CapturedAnthropicRequest): number | undefined => {
		const seconds = req.model.promptCache?.short;
		return seconds && seconds > 0 ? seconds * 1000 : undefined;
	};

	const computeDecision = (req: CapturedAnthropicRequest): StoppingDecision | undefined => {
		if (config.mode === "off") return undefined;
		const ttlMs = getModelTtlMs(req);
		if (ttlMs === undefined) return undefined;
		const schedule = getRefreshSchedule(ttlMs);
		if (!schedule) return undefined;
		if (phase === "streaming") {
			return {
				shouldWarm: true,
				continuationProb: 1,
				expectedSavings: 0,
				cutoffIdleMs: schedule.ttlMs,
				remainingPings: 1,
			};
		}
		if (promptTokens <= 0) return undefined;
		const economics = computeCacheEconomics(req.model, promptTokens, forceOneToken ? 1 : 0);
		if (!economics.available) return undefined;
		const elapsedIdle = Math.max(0, Date.now() - idleSince);
		return evaluateOptimalStopping(
			config,
			elapsedIdle,
			schedule.delayMs,
			schedule.ttlMs,
			economics.warmCost,
			economics.missCost,
		);
	};

	const scheduleNext = (ctx: ExtensionContext, epoch: number) => {
		clearTimers();
		stopReason = undefined;
		if (epoch !== sessionEpoch || config.mode === "off" || !snapshot) return;
		const currentSnapshot = snapshot;
		const ttlMs = getModelTtlMs(currentSnapshot);
		const schedule = ttlMs !== undefined ? getRefreshSchedule(ttlMs) : undefined;
		if (!schedule) {
			stopReason = "no cache TTL";
			return;
		}
		if (phase === "idle" && promptTokens > 0) {
			lastDecision = computeDecision(currentSnapshot);
			if (lastDecision && !lastDecision.shouldWarm) {
				stopWarming("past curve horizon");
				return;
			}
		}

		nextWarmAt = Date.now() + schedule.delayMs;
		refreshDeadlineAt = nextWarmAt + schedule.graceMs;
		warmTimer = setTimeout(() => void runWarm(ctx, currentSnapshot, epoch), schedule.delayMs);
		warmTimer.unref?.();
	};

	const runWarm = async (ctx: ExtensionContext, targetSnapshot: CapturedAnthropicRequest, epoch: number) => {
		warmTimer = undefined;
		if (
			epoch !== sessionEpoch ||
			config.mode === "off" ||
			snapshot !== targetSnapshot ||
			ctx.model?.id !== targetSnapshot.model.id
		) {
			stopWarming();
			return;
		}
		if (Date.now() > refreshDeadlineAt) {
			stopWarming("missed refresh window");
			return;
		}

		const decision = computeDecision(targetSnapshot);
		lastDecision = decision;
		if (phase === "idle" && !decision?.shouldWarm) {
			stopWarming("past curve horizon");
			return;
		}

		const controller = new AbortController();
		inFlight = controller;

		try {
			let result = await sendWarmPing(ctx.modelRegistry, targetSnapshot, controller.signal, forceOneToken ? 1 : 0);
			if (
				result.rejectedZeroMaxTokens &&
				!controller.signal.aborted &&
				epoch === sessionEpoch &&
				snapshot === targetSnapshot
			) {
				forceOneToken = true;
				result = await sendWarmPing(ctx.modelRegistry, targetSnapshot, controller.signal, 1);
			}
			if (inFlight !== controller || epoch !== sessionEpoch) return;
			inFlight = undefined;

			if (!result.ok || snapshot !== targetSnapshot) {
				stopWarming(result.errorMessage ?? "request failed");
				return;
			}
			if (result.usage) {
				// If the server wrote a new cache entry without reading, the prefix already expired.
				if (result.usage.cacheWrite > 0 && result.usage.cacheRead === 0) {
					stopWarming("cache expired");
					return;
				}
				warmCount++;
				totalWarmCost += result.usage.cost.total;
			}
			scheduleNext(ctx, epoch);
		} catch {
			if (inFlight === controller) inFlight = undefined;
			if (epoch === sessionEpoch) stopWarming("request error");
		}
	};

	const restoreConfigFromBranch = (ctx: ExtensionContext) => {
		let nextConfig: CacheExtenderConfig = DEFAULT_CONFIG;
		let nextSource: ConfigSource = "default";
		const flagValue = pi.getFlag("cache-ttl");
		if (typeof flagValue === "string" && flagValue.trim().length > 0) {
			const parsedFlag = parseCommandArgs(flagValue);
			if (parsedFlag.action === "set") {
				nextConfig = parsedFlag.config;
				nextSource = "flag";
			}
		}
		for (const entry of ctx.sessionManager.getBranch()) {
			if (entry.type === "custom" && entry.customType === ENTRY_TYPE && isValidConfig(entry.data)) {
				nextConfig = entry.data;
				nextSource = "session";
			}
		}
		config = nextConfig;
		configSource = nextSource;
	};

	const resetRuntimeState = () => {
		sessionEpoch++;
		clearTimers();
		snapshot = undefined;
		promptTokens = 0;
		phase = "idle";
		inActiveTurn = false;
		idleSince = 0;
		warmCount = 0;
		totalWarmCost = 0;
		lastDecision = undefined;
		stopReason = undefined;
	};

	pi.on("session_start", async (_event, ctx) => {
		resetRuntimeState();
		restoreConfigFromBranch(ctx);
	});

	pi.on("session_tree", async (_event, ctx) => {
		resetRuntimeState();
		restoreConfigFromBranch(ctx);
	});

	pi.on("session_shutdown", async () => {
		sessionEpoch++;
		stopWarming();
	});

	pi.on("session_compact", async () => {
		snapshot = undefined;
		promptTokens = 0;
		stopWarming("compacted");
	});

	pi.on("model_select", async (event) => {
		if (snapshot && (event.model.provider !== snapshot.model.provider || event.model.id !== snapshot.model.id)) {
			snapshot = undefined;
			promptTokens = 0;
			forceOneToken = false;
			stopWarming("model changed");
		}
	});

	pi.on("turn_start", async () => {
		inActiveTurn = true;
		phase = "streaming";
		idleSince = 0;
	});

	pi.on("turn_end", async () => {
		inActiveTurn = false;
	});

	pi.on("before_provider_request", async (event, ctx) => {
		if (config.mode === "off" || !inActiveTurn) return;
		if (!isReplayableAnthropicPayload(event.payload, ctx.model)) return;
		snapshot = {
			model: ctx.model,
			payload: structuredClone(event.payload) as Record<string, unknown>,
			capturedAt: Date.now(),
		};
		scheduleNext(ctx, sessionEpoch);
	});

	pi.on("message_end", async (event) => {
		if (event.message.role !== "assistant") return;
		const usage = event.message.usage;
		if (!usage) return;
		const total = usage.input + usage.cacheRead + usage.cacheWrite;
		if (total > 0) promptTokens = total;
		if (snapshot && phase === "idle") {
			lastDecision = computeDecision(snapshot);
			if (lastDecision && !lastDecision.shouldWarm) stopWarming("past curve horizon");
		}
	});

	pi.on("agent_settled", async () => {
		inActiveTurn = false;
		phase = "idle";
		idleSince = Date.now();
		if (!snapshot || config.mode === "off") return;
		lastDecision = computeDecision(snapshot);
		if (!lastDecision?.shouldWarm) {
			stopWarming("past curve horizon");
		}
	});

	pi.on("cache_warming_decision", async (_event, ctx) => {
		// Suppress pi's built-in warmer for Anthropic models while this extension is active.
		if (config.mode !== "off" && ctx.model?.api === "anthropic-messages") {
			return { action: "stop" };
		}
	});

	pi.registerCommand("cache-ttl", {
		description: "Configure cache return curve (e.g. /cache-ttl 12m 60m, 30m, off, reset)",
		getArgumentCompletions: (prefix) => {
			const presets = ["12m 60m", "15m 90m", "30m", "off", "reset"];
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
				if (config.mode === "off") {
					stopWarming();
					ctx.ui.notify("Cache extender: off (using pi built-in warmer)", "info");
					return;
				}
				if (snapshot) {
					if (phase === "idle" && idleSince === 0) idleSince = snapshot.capturedAt;
					scheduleNext(ctx, sessionEpoch);
				}
			}

			const activeModel = snapshot?.model ?? ctx.model;
			const lines: string[] = [];

			if (config.mode === "off") {
				lines.push(`Cache Extender: off (${configSource})`);
			} else {
				lines.push(
					`Cache Extender: ${formatCurveSummary(config)} (${configSource}) — 90% @ ${formatDuration(config.t90Ms)}, 50% @ ${formatDuration(config.t50Ms)}, 10% @ ${formatDuration(config.t10Ms)}, k=${config.k.toFixed(2)}`,
				);
			}

			let stateText = "idle";
			if (config.mode === "off") {
				stateText = "disabled";
			} else if (inFlight) {
				stateText = "warming now";
			} else if (warmTimer && nextWarmAt > 0) {
				const nextText = `next in ${formatDuration(Math.max(0, nextWarmAt - Date.now()))}`;
				if (snapshot && promptTokens > 0) {
					const d = computeDecision(snapshot) ?? lastDecision;
					stateText =
						d && d.remainingPings > 0
							? `${nextText} | horizon ~${formatDuration(d.cutoffIdleMs)} (${d.remainingPings} pings max)`
							: nextText;
				} else {
					stateText = nextText;
				}
			} else if (stopReason) {
				stateText = `stopped (${stopReason})`;
			}
			lines.push(`Status: ${stateText}`);

			if (activeModel) {
				if (promptTokens > 0) {
					const econ = computeCacheEconomics(activeModel, promptTokens, forceOneToken ? 1 : 0);
					const econText = econ.available
						? ` | warm ${formatCost(econ.warmCost)} vs miss ${formatCost(econ.missCost)}`
						: "";
					lines.push(`Model: ${activeModel.id} (${promptTokens.toLocaleString()} tokens${econText})`);
				} else {
					lines.push(`Model: ${activeModel.id}`);
				}
			}

			if (warmCount > 0) {
				lines.push(`Session warmed: ${warmCount}x (${formatCost(totalWarmCost)})`);
			}

			if (parsed.action === "status") {
				lines.push("Usage: /cache-ttl [t90 t10 | t50 | off | reset]");
			}

			ctx.ui.notify(lines.join("\n"), "info");
		},
	});
}
