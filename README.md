# pi-cache-extender

Keeps Anthropic prompt caches warm in [pi](https://pi.dev) for as long as it pays off. It replaces two things in pi's built-in cache warmer: the 30-minute idle cap, and the fixed 15% chance that you come back. Instead you describe when you tend to return with a 2-point curve (`t90` / `t10`), and the extension keeps refreshing the cache while the expected saving of a cache hit exceeds the cost of the refreshes.

A refresh replays the last request with a one-token output cap, shortly before the cache entry expires (5 minutes, or 1 hour with `PI_CACHE_RETENTION=long`). While an agent run is active, refreshes continue for up to an hour after its last request. Refresh costs are not included in pi's session totals; `/cache-ttl` shows them.

## Usage

Install or load the extension:

```bash
pi install ./pi-cache-extender
```

Configure the return curve per session with `/cache-ttl` or at startup via `--cache-ttl`:

```text
/cache-ttl 1.5m 3h   # 90% chance still needed at 1.5m, 10% at 3h (default)
/cache-ttl 30m       # 50% half-life at 30m
/cache-ttl off       # fall back to pi's built-in cache warmer
/cache-ttl reset     # restore the default
/cache-ttl           # show current curve, next refresh, and session warm cost
```
