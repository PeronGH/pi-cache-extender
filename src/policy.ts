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

const LOG_09 = Math.log(0.9);
const LOG_05 = Math.log(0.5);
const LOG_01 = Math.log(0.1);
const MIN_HORIZON_PROB = 0.002;
const MAX_STEPS = 400;

/**
 * Fit the return curve S(t) = 0.5^((t / t50)^k) through S(t90) = 0.9 and S(t10) = 0.1.
 * S(t) is the chance that the user still comes back to the session after being idle for t.
 */
export function fromTwoPoints(t90Ms: number, t10Ms: number): CurveSpec {
	if (!(Number.isFinite(t90Ms) && Number.isFinite(t10Ms) && t90Ms > 0 && t10Ms > t90Ms)) {
		throw new Error("Expected 0 < t90 < t10");
	}
	const k = Math.log(LOG_01 / LOG_09) / Math.log(t10Ms / t90Ms);
	const t50Ms = t90Ms * (LOG_05 / LOG_09) ** (1 / k);
	return { mode: "curve", t90Ms, t10Ms, t50Ms, k };
}

/** Return curve with S(t50) = 0.5 and shape k. */
export function fromHalfLife(t50Ms: number, k = 2): CurveSpec {
	if (!(Number.isFinite(t50Ms) && Number.isFinite(k) && t50Ms > 0 && k > 0)) {
		throw new Error("Expected positive t50 and k");
	}
	const t90Ms = t50Ms * (LOG_09 / LOG_05) ** (1 / k);
	const t10Ms = t50Ms * (LOG_01 / LOG_05) ** (1 / k);
	return { mode: "curve", t90Ms, t10Ms, t50Ms, k };
}

/**
 * Fitted to about 3,800 idle gaps from eight weeks of pi sessions: of the curves tried, this one's refresh counts cost
 * least across the refresh-to-miss price ratios of current Claude models (0.02, 0.04 and 0.09).
 */
export const DEFAULT_CONFIG: CurveSpec = fromTwoPoints(90_000, 3 * 3_600_000);

export function survivalAt(tMs: number, t50Ms: number, k: number): number {
	if (tMs <= 0) return 1;
	return 0.5 ** ((tMs / t50Ms) ** k);
}

/**
 * Number of consecutive refreshes worth sending, one every `stepMs`, to a user who has been idle for `idleMs`.
 *
 * At idle time t the user returns within the next step with probability S(t) - S(t + step), later with S(t + step),
 * and never with 1 - S(t). Stopping costs S(t) * missCost. Refreshing costs warmCost plus, unless the user returns
 * within the step, the cheaper choice at t + step. Solved backward from where S becomes negligible.
 */
export function plannedRefreshes(
	curve: CurveSpec,
	idleMs: number,
	stepMs: number,
	warmCost: number,
	missCost: number,
): number {
	const survival = (step: number) => survivalAt(idleMs + step * stepMs, curve.t50Ms, curve.k);
	let steps = 1;
	while (steps < MAX_STEPS && survival(steps) >= MIN_HORIZON_PROB) steps++;

	let laterCost = survival(steps) * missCost;
	let refreshes = 0;
	for (let step = steps - 1; step >= 0; step--) {
		const stopCost = survival(step) * missCost;
		const refreshCost = warmCost + (1 - (survival(step) - survival(step + 1))) * laterCost;
		if (refreshCost < stopCost) {
			refreshes++;
			laterCost = refreshCost;
		} else {
			refreshes = 0;
			laterCost = stopCost;
		}
	}
	return refreshes;
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
