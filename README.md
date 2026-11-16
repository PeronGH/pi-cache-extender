# pi-cache-extender

Extends Anthropic prompt cache lifetimes in [pi](https://pi.dev) using a 2-point return probability curve (`t90` / `t10`) and zero-output (`max_tokens: 0`) background refreshes.

## Usage

Install or load the extension:

```bash
pi install ./pi-cache-extender
```

Configure the return curve per session with `/cache-ttl` or at startup via `--cache-ttl`:

```text
/cache-ttl 12m 60m   # 90% chance still needed at 12m, 10% at 60m (default)
/cache-ttl 30m       # 50% half-life at 30m
/cache-ttl off       # fall back to pi's built-in cache warmer
/cache-ttl reset     # restore 12m -> 60m default
/cache-ttl           # show current curve, horizon, and session warm cost
```
