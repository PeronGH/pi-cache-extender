import type { Api, Context, Model, Usage } from "@earendil-works/pi-ai";
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";

export interface CapturedAnthropicRequest {
	model: Model<"anthropic-messages">;
	payload: Record<string, unknown>;
	capturedAt: number;
}

export interface WarmPingResult {
	ok: boolean;
	usage?: Usage;
	rejectedZeroMaxTokens?: boolean;
	errorMessage?: string;
}

function hasCacheControlValue(value: unknown): boolean {
	if (!value || typeof value !== "object") return false;
	if (Array.isArray(value)) return value.some(hasCacheControlValue);
	const record = value as Record<string, unknown>;
	if (record.cache_control && typeof record.cache_control === "object") return true;
	return Object.values(record).some(hasCacheControlValue);
}

/** Verify the payload belongs to a cacheable Anthropic model and can be replayed with max_tokens: 0. */
export function isReplayableAnthropicPayload(
	payload: unknown,
	model: Model<Api> | undefined,
): model is Model<"anthropic-messages"> {
	if (model?.api !== "anthropic-messages") return false;
	if (!model.promptCache?.short || model.promptCache.short <= 0) return false;
	if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
	const record = payload as Record<string, unknown>;
	if (record.model !== model.id || !Array.isArray(record.messages)) return false;
	// Budget-based extended thinking requires budget_tokens < max_tokens and cannot use max_tokens: 0.
	const thinking = record.thinking;
	if (thinking && typeof thinking === "object" && (thinking as Record<string, unknown>).type === "enabled") {
		return false;
	}
	return (
		hasCacheControlValue(record.system) || hasCacheControlValue(record.tools) || hasCacheControlValue(record.messages)
	);
}

/** Clone a captured provider payload for a streaming zero-output warm replay via pi's SDK pipeline. */
export function buildWarmPayload(payload: Record<string, unknown>, maxTokens = 0): Record<string, unknown> {
	const next = structuredClone(payload);
	delete next.thread;
	next.max_tokens = maxTokens;
	next.stream = true;
	return next;
}

export async function sendWarmPing(
	modelRegistry: Pick<ModelRegistry, "streamSimple">,
	snapshot: CapturedAnthropicRequest,
	signal: AbortSignal,
	maxTokens = 0,
): Promise<WarmPingResult> {
	const dummyContext: Context = {
		messages: [{ role: "user", content: "ping", timestamp: Date.now() }],
	};
	const message = await modelRegistry
		.streamSimple(snapshot.model, dummyContext, {
			maxTokens: 1,
			maxRetries: 0,
			disableContinuation: true,
			signal,
			onPayload: () => buildWarmPayload(snapshot.payload, maxTokens),
		})
		.result();

	if (message.stopReason === "error" || message.stopReason === "aborted") {
		const errorMessage = message.errorMessage ?? message.stopReason;
		return {
			ok: false,
			rejectedZeroMaxTokens: maxTokens === 0 && /max_tokens/i.test(errorMessage),
			errorMessage,
		};
	}

	return {
		ok: true,
		usage: message.usage,
	};
}
