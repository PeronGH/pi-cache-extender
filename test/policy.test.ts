import { describe, expect, test } from "bun:test";
import { evaluateCacheWarming, type Model } from "@earendil-works/pi-ai";
import {
	DEFAULT_CONFIG,
	formatCurveSummary,
	formatDuration,
	fromHalfLife,
	fromTwoPoints,
	parseCommandArgs,
	parseDurationMs,
	plannedRefreshes,
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
		expect(formatCurveSummary(DEFAULT_CONFIG)).toBe("1m30s -> 3h");
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

describe("plannedRefreshes", () => {
	const STEP_MS = 270_000;
	const planFor = (model: Model<"anthropic-messages">, idleMs: number) => {
		const { warmCost, missCost } = evaluateCacheWarming(model, 50_000, "idle");
		return plannedRefreshes(DEFAULT_CONFIG, idleMs, STEP_MS, warmCost, missCost);
	};

	test("keeps a cache warm longer the more a miss costs relative to a refresh", () => {
		const fable = planFor(makeModel(), STEP_MS);
		const opus = planFor(makeModel({ cost: { input: 4, output: 20, cacheRead: 0.2, cacheWrite: 5 } }), STEP_MS);
		const haiku = planFor(
			makeModel({ cost: { input: 0.1, output: 0.5, cacheRead: 0.01, cacheWrite: 0.125 } }),
			STEP_MS,
		);

		expect(haiku).toBeGreaterThan(0);
		expect(opus).toBeGreaterThan(haiku);
		expect(fable).toBeGreaterThan(opus);
	});

	test("plans nothing once past the curve tail", () => {
		expect(planFor(makeModel(), 90 * 60_000)).toBe(0);
	});
});
