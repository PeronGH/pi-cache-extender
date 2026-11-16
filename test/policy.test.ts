import { describe, expect, test } from "bun:test";
import type { Model } from "@earendil-works/pi-ai";
import {
	computeCacheEconomics,
	DEFAULT_CONFIG,
	evaluateOptimalStopping,
	formatCurveSummary,
	formatDuration,
	fromHalfLife,
	fromTwoPoints,
	getRefreshSchedule,
	parseCommandArgs,
	parseDurationMs,
	survivalAt,
} from "../src/policy.ts";

function makeModel(overrides: Partial<Model<"anthropic-messages">> = {}): Model<"anthropic-messages"> {
	return {
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
		...overrides,
	};
}

describe("curve fitting", () => {
	test("fromTwoPoints hits exact 90% and 10% quantiles", () => {
		const curve = fromTwoPoints(12 * 60_000, 60 * 60_000);
		expect(survivalAt(12 * 60_000, curve.t50Ms, curve.k)).toBeCloseTo(0.9, 6);
		expect(survivalAt(curve.t50Ms, curve.t50Ms, curve.k)).toBeCloseTo(0.5, 6);
		expect(survivalAt(60 * 60_000, curve.t50Ms, curve.k)).toBeCloseTo(0.1, 6);
	});

	test("fromHalfLife matches DEFAULT_CONFIG scale", () => {
		const curve = fromHalfLife(30 * 60_000);
		expect(curve.k).toBe(2);
		expect(survivalAt(30 * 60_000, curve.t50Ms, curve.k)).toBeCloseTo(0.5, 6);
		expect(survivalAt(curve.t90Ms, curve.t50Ms, curve.k)).toBeCloseTo(0.9, 6);
		expect(survivalAt(curve.t10Ms, curve.t50Ms, curve.k)).toBeCloseTo(0.1, 6);
	});

	test("rejects invalid point ordering", () => {
		expect(() => fromTwoPoints(60_000, 60_000)).toThrow();
		expect(() => fromTwoPoints(120_000, 60_000)).toThrow();
	});
});

describe("command and duration parsing", () => {
	test("parses durations and formats cleanly", () => {
		expect(parseDurationMs("30s")).toBe(30_000);
		expect(parseDurationMs("12m")).toBe(720_000);
		expect(parseDurationMs("1.5h")).toBe(5_400_000);
		expect(parseDurationMs("15")).toBe(900_000);
		expect(parseDurationMs("0m")).toBeUndefined();
		expect(formatDuration(45_000)).toBe("45s");
		expect(formatDuration(270_000)).toBe("4m30s");
		expect(formatDuration(3_600_000)).toBe("1h");
		expect(formatCurveSummary(DEFAULT_CONFIG)).toBe("12m -> 1h");
	});

	test("parses /cache-ttl arguments", () => {
		expect(parseCommandArgs("")).toEqual({ action: "status" });
		expect(parseCommandArgs("off")).toEqual({ action: "set", config: { mode: "off" } });
		expect(parseCommandArgs("reset")).toEqual({ action: "set", config: DEFAULT_CONFIG });

		const one = parseCommandArgs("30m");
		expect(one.action).toBe("set");
		if (one.action === "set" && one.config.mode === "curve") {
			expect(one.config.t50Ms).toBe(30 * 60_000);
		}

		const two = parseCommandArgs("10m 1h");
		expect(two.action).toBe("set");
		if (two.action === "set" && two.config.mode === "curve") {
			expect(two.config.t90Ms).toBe(10 * 60_000);
			expect(two.config.t10Ms).toBe(60 * 60_000);
		}

		expect(parseCommandArgs("60m 10m").action).toBe("invalid");
	});
});

describe("optimal stopping DP", () => {
	test("keeps Fable 5.2 warm longer than Opus 5.5 and Haiku 5.5 under identical curve", () => {
		const schedule = getRefreshSchedule(300_000);
		expect(schedule).toBeDefined();
		if (!schedule) return;
		const fable = makeModel({ cost: { input: 8, output: 40, cacheRead: 0.2, cacheWrite: 10 } });
		const opus = makeModel({ id: "claude-opus-5-5", cost: { input: 4, output: 20, cacheRead: 0.2, cacheWrite: 5 } });
		const haiku = makeModel({
			id: "claude-haiku-5-5",
			cost: { input: 0.1, output: 0.5, cacheRead: 0.01, cacheWrite: 0.125 },
		});

		const evalFor = (model: Model<"anthropic-messages">) => {
			const econ = computeCacheEconomics(model, 50_000, 0);
			return evaluateOptimalStopping(
				DEFAULT_CONFIG,
				4.5 * 60_000,
				schedule.delayMs,
				schedule.ttlMs,
				econ.warmCost,
				econ.missCost,
			);
		};

		const dFable = evalFor(fable);
		const dOpus = evalFor(opus);
		const dHaiku = evalFor(haiku);

		expect(dFable.shouldWarm).toBe(true);
		expect(dOpus.shouldWarm).toBe(true);
		expect(dHaiku.shouldWarm).toBe(true);
		expect(dFable.remainingPings).toBeGreaterThan(dOpus.remainingPings);
		expect(dOpus.remainingPings).toBeGreaterThan(dHaiku.remainingPings);
	});

	test("stops once past the curve tail", () => {
		const schedule = getRefreshSchedule(300_000);
		expect(schedule).toBeDefined();
		if (!schedule) return;
		const fable = makeModel();
		const econ = computeCacheEconomics(fable, 50_000, 0);
		const late = evaluateOptimalStopping(
			DEFAULT_CONFIG,
			90 * 60_000,
			schedule.delayMs,
			schedule.ttlMs,
			econ.warmCost,
			econ.missCost,
		);
		expect(late.shouldWarm).toBe(false);
		expect(late.remainingPings).toBe(0);
	});
});
