import { type Api, type AssistantMessage, getCacheWarmingDelayMs, type Model } from "@earendil-works/pi-ai";
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";

/** A request whose prompt cache entry can be refreshed by replaying its payload. */
export interface CapturedRequest {
	model: Model<"anthropic-messages">;
	payload: Record<string, unknown>;
	/** Lifetime of the cache entry the request wrote. */
	ttlMs: number;
	/** How long after a cache touch the next refresh is due. */
	delayMs: number;
}

function findCacheControl(value: unknown): Record<string, unknown> | undefined {
	if (!value || typeof value !== "object") return undefined;
	const cacheControl = (value as Record<string, unknown>).cache_control;
	if (cacheControl && typeof cacheControl === "object") return cacheControl as Record<string, unknown>;
	for (const child of Object.values(value)) {
		const found = findCacheControl(child);
		if (found) return found;
	}
	return undefined;
}

/** Capture an agent request for replay, or undefined when a one-token replay cannot refresh its cache entry. */
export function captureRequest(payload: unknown, model: Model<Api> | undefined): CapturedRequest | undefined {
	if (model?.api !== "anthropic-messages" || !payload || typeof payload !== "object") return undefined;
	const record = payload as Record<string, unknown>;
	if (record.model !== model.id) return undefined;
	// Budget-based thinking requires budget_tokens < max_tokens, so it rejects a one-token replay.
	if ((record.thinking as { type?: unknown } | undefined)?.type === "enabled") return undefined;
	// pi puts the same cache_control on every breakpoint, so the first one tells the retention tier.
	const cacheControl = findCacheControl(record);
	if (!cacheControl) return undefined;
	const seconds = model.promptCache?.[cacheControl.ttl === "1h" ? "long" : "short"];
	if (!seconds) return undefined;
	const ttlMs = seconds * 1000;
	const delayMs = getCacheWarmingDelayMs(ttlMs);
	if (delayMs === undefined) return undefined;
	return { model: model as Model<"anthropic-messages">, payload: structuredClone(record), ttlMs, delayMs };
}

/** Replay a captured request with a one-token output cap. The context is a placeholder; the payload is replaced. */
export async function sendWarmPing(
	modelRegistry: Pick<ModelRegistry, "streamSimple">,
	request: CapturedRequest,
	signal: AbortSignal,
): Promise<AssistantMessage> {
	return modelRegistry
		.streamSimple(
			request.model,
			{ messages: [{ role: "user", content: "ping", timestamp: Date.now() }] },
			{ maxRetries: 0, signal, onPayload: () => ({ ...request.payload, max_tokens: 1 }) },
		)
		.result();
}
