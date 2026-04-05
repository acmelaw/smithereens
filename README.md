# smithereens

An experimental lightweight LLM interface.

[![CI](https://github.com/acmelaw/smithereens/actions/workflows/ci.yml/badge.svg)](https://github.com/acmelaw/smithereens/actions/workflows/ci.yml)

## Quick start

```bash
# Install with PDM
pdm install

# Run interactively
pdm run smithereens

# One-shot prompt
pdm run smithereens "explain this codebase"
```

## Development

```bash
# Install all dependencies (dev + test)
make install

# Run linter and formatter
make lint
make format

# Run tests
make test

# Run all checks (lint + format + test)
make check
```

## Container

Build and run using the [Containerfile](Containerfile) (OCI-compliant, multi-stage build):

```bash
# Build the image
make build

# Run interactively via compose
make run

# Or directly with docker
docker build -f Containerfile -t smithereens .
docker run -it --rm \
  -e OPENAI_API_KEY \
  -v "$PWD:/workspace" \
  smithereens
```

The [compose.yaml](compose.yaml) handles volume mounts and environment variable
passthrough for API keys.

## Configuration

Smithereens reads `~/.smithereens/config.toml`:

```toml
model = "ollama_chat/gemma4:26b"
max_tokens = 16384
permissions = "ask"   # or "allow"
```

Override via environment variables:

| Variable | Description |
|---|---|
| `SMITHEREENS_MODEL` | LLM model to use |
| `SMITHEREENS_MAX_TOKENS` | Max tokens per response |
| `SMITHEREENS_PERMISSIONS` | `ask`, `allow`, or `deny` |

## Acknowledgements

Inspired by [claude.sh](https://github.com/jdcodes1/claude-sh).

## License

[MIT](LICENSE.md)