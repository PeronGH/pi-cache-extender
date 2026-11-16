import { type Api, calculateCost, type Model, type Usage } from "@earendil-works/pi-ai";

export interface CurveSpec {
	mode: "curve";
	t90Ms: number;
	t10Ms: number;
	t50Ms: number;
	k: number;
}

export type CacheExtenderConfig = { mode: "off" } | CurveSpec;

export type ParsedCommand =
	| { action: "status" }
	| { action: "set"; config: CacheExtenderConfig }
	| { action: "invalid"; message: string };

export interface RefreshSchedule {
	ttlMs: number;
	delayMs: number;
	graceMs: number;
}

export interface StoppingDecision {
	shouldWarm: boolean;
	continuationProb: number;
	expectedSavings: number;
	cutoffIdleMs: number;
	remainingPings: number;
}

const LOG_09 = Math.log(0.9);
const LOG_05 = Math.log(0.5);
const LOG_01 = Math.log(0.1);
const MIN_HORIZON_PROB = 0.002;
const MAX_DP_STEPS = 400;

/** Fit a Weibull survival curve S(t) = 0.5^((t / t50)^k) from 90% and 10% return times. */
export function fromTwoPoints(t90Ms: number, t10Ms: number): CurveSpec {
	if (!(Number.isFinite(t90Ms) && Number.isFinite(t10Ms) && t90Ms > 0 && t10Ms > t90Ms)) {
		throw new Error("Expected 0 < t90 < t10");
	}
	const k = Math.log(LOG_01 / LOG_09) / Math.log(t10Ms / t90Ms);
	const t50Ms = t90Ms * (LOG_05 / LOG_09) ** (1 / k);
	return { mode: "curve", t90Ms, t10Ms, t50Ms, k };
}

/** Fit a Weibull survival curve from half-life t50 and shape k (defaults to 2). */
export function fromHalfLife(t50Ms: number, k = 2): CurveSpec {
	if (!(Number.isFinite(t50Ms) && Number.isFinite(k) && t50Ms > 0 && k > 0)) {
		throw new Error("Expected positive t50 and k");
	}
	const t90Ms = t50Ms * (LOG_09 / LOG_05) ** (1 / k);
	const t10Ms = t50Ms * (LOG_01 / LOG_05) ** (1 / k);
	return { mode: "curve", t90Ms, t10Ms, t50Ms, k };
}

export const DEFAULT_CONFIG: CurveSpec = fromTwoPoints(12 * 60_000, 60 * 60_000);

export function survivalAt(tMs: number, t50Ms: number, k: number): number {
	if (tMs <= 0) return 1;
	return 0.5 ** ((tMs / t50Ms) ** k);
}

/** Match pi's 90% / -10s pre-expiry timing with half the remaining margin as sleep grace. */
export function getRefreshSchedule(ttlMs: number): RefreshSchedule | undefined {
	if (ttlMs <= 10_000) return undefined;
	const delayMs = Math.max(1, Math.floor(Math.min(ttlMs * 0.9, ttlMs - 10_000)));
	const graceMs = Math.floor((ttlMs - delayMs) / 2);
	return { ttlMs, delayMs, graceMs };
}

/** Compute single-refresh cost and cache-miss penalty using pi-ai's tiered calculateCost. */
export function computeCacheEconomics(
	model: Model<Api>,
	promptTokens: number,
	warmOutputTokens = 0,
): { warmCost: number; missCost: number; available: boolean } {
	if (promptTokens <= 0) return { warmCost: 0, missCost: 0, available: false };
	const price = (tokens: Partial<Pick<Usage, "input" | "output" | "cacheRead" | "cacheWrite">>): number => {
		const usage: Usage = {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			...tokens,
		};
		return calculateCost(model, usage).total;
	};
	const cacheHitCost = price({ cacheRead: promptTokens });
	const cacheMissCost = price(model.cost.cacheWrite > 0 ? { cacheWrite: promptTokens } : { input: promptTokens });
	const warmCost = price({ cacheRead: promptTokens, output: warmOutputTokens });
	const missCost = Math.max(0, cacheMissCost - cacheHitCost);
	return { warmCost, missCost, available: cacheHitCost > 0 || cacheMissCost > 0 };
}

/**
 * Backward dynamic programming optimal stopping rule over the Weibull curve.
 * Compares stopping now (losing q(t) * missCost) against warming now and acting optimally thereafter.
 */
export function evaluateOptimalStopping(
	config: CurveSpec,
	elapsedIdleMs: number,
	stepMs: number,
	ttlMs: number,
	warmCost: number,
	missCost: number,
): StoppingDecision {
	const start = Math.max(0, elapsedIdleMs);
	const q0 = survivalAt(start, config.t50Ms, config.k);
	if (missCost <= 0 || warmCost < 0 || stepMs <= 0) {
		return {
			shouldWarm: false,
			continuationProb: q0,
			expectedSavings: 0,
			cutoffIdleMs: start,
			remainingPings: 0,
		};
	}

	let steps = 1;
	while (steps < MAX_DP_STEPS && survivalAt(start + steps * stepMs, config.t50Ms, config.k) >= MIN_HORIZON_PROB) {
		steps++;
	}

	const warmFlags = new Array<boolean>(steps);
	let nextValue = survivalAt(start + steps * stepMs, config.t50Ms, config.k) * missCost;
	let rootSavings = 0;

	for (let i = steps - 1; i >= 0; i--) {
		const t = start + i * stepMs;
		const q = survivalAt(t, config.t50Ms, config.k);
		const qNext = survivalAt(t + stepMs, config.t50Ms, config.k);
		const pReturn = Math.max(0, q - qNext);
		const costStop = q * missCost;
		const costWarm = warmCost + (1 - pReturn) * nextValue;
		const warm = costWarm < costStop;
		warmFlags[i] = warm;
		nextValue = warm ? costWarm : costStop;
		if (i === 0) rootSavings = costStop - costWarm;
	}

	let remainingPings = 0;
	for (let i = 0; i < steps && warmFlags[i]; i++) {
		remainingPings++;
	}
	const cutoffIdleMs = remainingPings > 0 ? start + (remainingPings - 1) * stepMs + ttlMs : start;

	return {
		shouldWarm: warmFlags[0] ?? false,
		continuationProb: q0,
		expectedSavings: rootSavings,
		cutoffIdleMs,
		remainingPings,
	};
}

export function parseDurationMs(token: string): number | undefined {
	const match = /^(\d+(?:\.\d+)?)(s|m|h)?$/i.exec(token.trim());
	if (!match) return undefined;
	const value = Number(match[1]);
	if (!Number.isFinite(value) || value <= 0) return undefined;
	const unit = (match[2] ?? "m").toLowerCase();
	const multiplier = unit === "h" ? 3_600_000 : unit === "s" ? 1_000 : 60_000;
	return Math.round(value * multiplier);
}

export function formatDuration(ms: number): string {
	const totalSec = Math.max(0, Math.round(ms / 1000));
	if (totalSec < 60) return `${totalSec}s`;
	const hours = Math.floor(totalSec / 3600);
	const mins = Math.floor((totalSec % 3600) / 60);
	const secs = totalSec % 60;
	if (hours > 0) {
		return mins > 0 ? `${hours}h${mins}m` : `${hours}h`;
	}
	return secs > 0 && mins < 10 ? `${mins}m${secs}s` : `${mins}m`;
}

export function formatCurveSummary(config: CacheExtenderConfig): string {
	if (config.mode === "off") return "off";
	return `${formatDuration(config.t90Ms)} -> ${formatDuration(config.t10Ms)}`;
}

export function parseCommandArgs(raw: string): ParsedCommand {
	const trimmed = raw.trim();
	if (!trimmed) return { action: "status" };
	const parts = trimmed.split(/\s+/);
	if (parts.length === 1) {
		const word = parts[0].toLowerCase();
		if (word === "off") return { action: "set", config: { mode: "off" } };
		if (word === "reset" || word === "default") return { action: "set", config: DEFAULT_CONFIG };
		const t50 = parseDurationMs(parts[0]);
		if (t50 !== undefined) return { action: "set", config: fromHalfLife(t50) };
		return { action: "invalid", message: "Usage: /cache-ttl [t90 t10 | t50 | off | reset]" };
	}
	if (parts.length === 2) {
		const t90 = parseDurationMs(parts[0]);
		const t10 = parseDurationMs(parts[1]);
		if (t90 === undefined || t10 === undefined || t10 <= t90) {
			return { action: "invalid", message: "Expected two durations with t90 < t10 (e.g. 12m 60m)" };
		}
		return { action: "set", config: fromTwoPoints(t90, t10) };
	}
	return { action: "invalid", message: "Usage: /cache-ttl [t90 t10 | t50 | off | reset]" };
}
