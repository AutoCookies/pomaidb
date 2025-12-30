# pomai-cache (Go core)

This file documents how to clone, build, run and contribute to the Go core of pomai-cache. It is intended for developers who want to run the server locally, inspect core behavior, and open contributions.

Note: this document focuses on the Go service (core) — not the TypeScript SDK. If you want to test or develop the SDK, see the separate SDK docs. Here: https://www.npmjs.com/package/@autocookie/pomai-cache

---

## Prerequisites

- Git
- Go (recommended >= 1.20). The repository uses Go Modules.
- Docker & Docker Compose (recommended for running a local DB and the full stack)
- Optional: make, dlv (debugger), and common tools (golangci-lint, gofmt)

---

## Quickstart — clone & branch

1. Clone the repository:

```bash
git clone https://github.com/AutoCookies/pomai-cache.git
cd pomai-cache
```

2. Create a new feature branch (follow your team naming convention):

```bash
# Fetch latest and create branch from main
git checkout main
git pull origin main
git checkout -b feat/<short-description>
```

Branch name examples:
- `feat/ttl-policy`
- `fix/discovery-timeout`
- `docs/contributing`

---

## Dependencies (Go modules)

This repository uses Go modules. To download dependencies:

```bash
# download required modules
go mod download

# alternatively, when adding a dependency:
go get github.com/some/dependency@vX.Y.Z
```

Notes:
- Avoid using `GOPATH`-dependent workflows — prefer modules mode.
- If you edit go.mod, run `go mod tidy` to keep it clean.

---

## Environment configuration

Place runtime configuration in an `.env` file (or export env vars). Use `.env.example` as a starting point. Example `.env.example` (copy to `.env` then edit):

```env
# .env.example
# Server
POMAI_PORT=8080
POMAI_ENV=development

# Database (Postgres example)
POMAI_DB_DSN=postgres://postgres:password@127.0.0.1:5432/pomaidb?sslmode=disable

# API / secrets (placeholder)
POMAI_API_KEY=your_api_key_here
POMAI_SECRET=some-secret-value

# Discovery / service addresses (optional)
POMAI_DISCOVERY_URL=http://127.0.0.1:8080
```

Copy and edit:

```bash
cp .env.example .env
# edit .env with your values
```

Important: do not commit `.env` with real secrets. Keep `.env` in `.gitignore`.

---

## Database setup

This project contains `init.sql` (schema or initial data). There are two common ways to initialize the database:

1. Using Docker Compose (recommended for local development):

- If the repo includes `docker-compose.yaml` you can start the DB:

```bash
docker compose up -d
# wait until DB is ready, then run migrations or init script
```

- Apply the initialization SQL (example):

```bash
# run init.sql against the containerized Postgres instance
docker exec -i <postgres_container_name> psql $POMAI_DB_DSN < init.sql
```

2. Manual DB setup:

- Create a Postgres instance locally and run `init.sql` using psql or a DB client.

---

## Build and run

From repository root:

```bash
# build the binary
go build ./...

# run (reads env vars)
go run ./cmd/<server-main-package>
# or, if the main package is at cmd/server:
go run ./cmd/server/main.go
```

Replace the path above with the actual `cmd` entrypoint directory if different (check `./cmd`).

You can also run the built binary:

```bash
./pomaicache-server  # or the built binary name
```

If you prefer to run inside Docker, use the provided `Dockerfile` and `docker-compose.yaml`.

---

## Tests

Run unit tests:

```bash
go test ./... -v
```

Notes:
- For tests that require DB access, either run a local test DB or use test containers / mocks.
- Keep tests fast and deterministic; prefer mocking I/O in unit tests.

---

## Linting & formatting

- Format code with gofmt:
  ```bash
  gofmt -w .
  ```
- Run static analysis / linters (if configured):
  ```bash
  # example with golangci-lint
  golangci-lint run
  ```

---

## Common workflows

- Add a new dependency:
  ```bash
  go get github.com/example/dependency@latest
  go mod tidy
  git add go.mod go.sum
  git commit -m "chore(deps): add example dependency"
  ```

- Run with longer timeouts for debugging:
  Set `POMAI_TIMEOUT_MS` or increase relevant env variables in `.env`.

- Debug with Delve:
  ```bash
  dlv debug ./cmd/server -- -config=./config.yaml
  ```

---

## Contributing (short)

1. Fork the repo and create a branch:
   ```bash
   git checkout -b fix/some-bug
   ```

2. Make changes, run tests and linters:
   ```bash
   go test ./...
   gofmt -w .
   ```

3. Commit using Conventional Commits style:
   ```
   fix(discovery): handle discovery timeout
   ```

4. Push branch and open a PR against `AutoCookies/pomai-cache:main`. In your PR include:
   - Short summary
   - What problem it fixes or why change is needed
   - How to test
   - Any breaking changes or upgrade notes

---

## Opening a Pull Request

When your branch is ready:

- Push it to your fork:
  ```bash
  git push origin fix/some-bug
  ```

- Open a PR on GitHub:
  - Base: `AutoCookies/pomai-cache:main`
  - Head: `your-username:fix/some-bug`
  - Fill the PR template (description, steps to reproduce, tests)

- Expect CI to run (if configured). Address CI failures or requested changes.

---

## Security & secrets

- Never commit API keys, secrets, or production credentials.
- Use `.env.example` to document required env vars with placeholder values.
- For production deployments, use secret managers / environment injection (not .env files).

---

## Additional notes

- If you need to run the TypeScript SDK examples, consult the SDK README in the `pkg` or `pomaicacheui` directories (if present).
- If you are unsure where the server entrypoint is, look under the `cmd/` folder.
- If the project uses migrations or a migration tool (e.g., `migrate`), prefer that workflow for schema changes rather than raw SQL files.

---

If you want, I can:
- Produce a ready-to-commit `.env.example` file content for this repo,
- Create a short CONTRIBUTING snippet targeted for Go core only,
- Or prepare the exact `go run` command by inspecting the `cmd/` layout (tell me where the main package lives).
Which would you like next?