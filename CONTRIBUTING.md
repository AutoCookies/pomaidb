# Contributing to pomai-cache

Thank you for your interest in contributing to pomai-cache — we appreciate help from the community. This document explains how to report bugs, propose features, set up a local development environment, run tests and examples, and prepare a high-quality pull request.

If you're unsure about the scope or impact of your change, please open an issue first to discuss it.

---

## Table of contents

- Code of conduct
- Where to start
- Reporting bugs
- Requesting features
- Development workflow (fork → branch → PR)
- Branch naming and commit guidelines
- Tests, linting and type checking
- Running examples locally
- Documentation changes
- Pull request checklist
- Review & merge process
- Release & versioning notes
- Thanks

---

## Code of conduct

Be respectful, kind, and collaborative. By contributing you agree to follow the project's Code of Conduct. If the repository does not include one yet, we recommend adopting the [Contributor Covenant](https://www.contributor-covenant.org/). Report incidents by opening an issue or contacting the maintainers directly.

---

## Where to start

- Read the repository `README.md` to understand the project's goals and public API.
- Look at issues labeled `good first issue` or `help wanted` for ways to help.
- Browse the `examples/` folder to see how the SDK is used in practice.

---

## Reporting bugs

Before opening a bug report:
1. Search existing issues to avoid duplicates.
2. Try to reproduce the issue against the latest `main` branch.

When filing a bug, include:
- A clear description of the problem and expected behavior.
- Minimal code to reproduce the issue (TypeScript/JavaScript snippet).
- Environment details: Node.js version, OS, TypeScript version (if relevant), and pomai-cache version.
- Any relevant logs, HTTP request/response snippets (remove API keys), and stack traces.

Use the bug report issue template when creating the issue.

---

## Requesting features

When suggesting a feature:
- Explain the problem you're solving and why it's useful.
- Suggest a concrete API or configuration example if relevant.
- Describe any backwards-compatibility considerations and migration steps.

Use the feature request issue template to provide consistent, actionable requests.

---

## Development workflow

Recommended workflow:

1. Fork the repo.
2. Clone your fork and add the upstream remote:
   ```bash
   git clone git@github.com:<your-username>/pomai-cache.git
   cd pomai-cache
   git remote add upstream https://github.com/AutoCookies/pomai-cache.git
   ```
3. Create a new branch from `main`:
   ```bash
   git fetch upstream
   git checkout -b feat/short-description upstream/main
   ```
   Branch prefixes:
   - feat/ — new features
   - fix/ — bug fixes
   - docs/ — documentation only
   - test/ — tests
   - chore/ — tasks, tooling, formatting

4. Make focused commits for each logical change.

5. Rebase or merge from upstream `main` frequently to keep your branch up to date:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

6. Push your branch to your fork and open a Pull Request against `AutoCookies/pomai-cache:main`.

---

## Branch naming and commit messages

- Use short, descriptive branch names:
  - `feat/ttl-api`, `fix/discovery-timeout`, `docs/readme-improvements`
- Follow the Conventional Commits style for commit messages:
  - `feat(scope): short description`
  - `fix(scope): short description`
  - `docs: update README`
  - Add longer description in the commit body when needed.
- If the change includes breaking changes, include `BREAKING CHANGE:` in the commit body with migration notes.

Example:
```
feat(example): add diagnostic example with fetch polyfill

BREAKING CHANGE: example import path changed to `../index`
```

---

## Tests, linting and type checking

- Add unit tests for new behaviors and bug fixes.
- Keep tests deterministic. Prefer mocking network calls instead of hitting real endpoints.
- Ensure TypeScript type checks pass:
  ```bash
  npx tsc --noEmit
  ```
- Run the repository's test command (if present):
  ```bash
  npm test
  ```
- If repository uses linters/formatters, run them before opening a PR:
  ```bash
  npm run lint
  npm run format
  ```
- If adding new dependencies, ensure they are necessary and lightweight.

---

## Running examples locally

The repository includes example(s) to demonstrate SDK usage. Recommended approach:

1. Install dependencies at repository root:
   ```bash
   npm install
   npm install -D ts-node typescript @types/node dotenv
   ```
2. Use a relative import in `examples/example.ts` so the example uses local source:
   ```ts
   import { PomaiClient } from '../index';
   ```
3. Create `examples/.env`:
   ```
   API_KEY=your_api_key_here
   ```
4. Run the example:
   ```bash
   cd examples
   npx ts-node -r dotenv/config example.ts
   ```

Notes:
- The SDK uses the global `fetch` API. For Node >= 18 this is built-in. For Node < 18, use a polyfill such as `node-fetch@2` or `undici`, and require it before running the example.

---

## Documentation changes

- Update `README.md` or add new docs when public APIs or behaviors change.
- Examples should be runnable and reflect the recommended usage.
- Keep changelog or release notes up to date for user-facing changes.

---

## Pull request checklist

Before requesting a review, ensure your PR contains:
- [ ] A descriptive title following Conventional Commits.
- [ ] A clear description of the change and motivation.
- [ ] Reference to related issue(s) (e.g., "Fixes #123").
- [ ] Tests that cover new behavior or fix the reported issue.
- [ ] Documentation updates if the public API or behavior changed.
- [ ] TypeScript builds/type checks without errors.
- [ ] No included secrets or API keys.
- [ ] CI checks pass (if CI is configured for the repo).

Suggested PR description template:
```
## Summary

One-paragraph summary of the change.

## Changes

- fileA.ts: Short note about the change
- fileB.md: Documentation updates

## Tests

- Added unit test for scenario X
- Updated mock for Y

## Related

- Fixes #<issue-number>
```

---

## Review & merge process

- Maintainers will review PRs and may request changes or tests.
- Small changes may be merged quickly; significant changes may require design discussion or progressive rollout.
- Respect reviewer feedback and update the branch accordingly.

---

## Release & versioning notes

- Releases should follow semantic versioning.
- Document breaking changes and migration steps in changelog or release notes.
- Coordinate with maintainers for major or breaking releases.

---

## Thanks

Thanks for taking the time to contribute! Your improvements — big or small — help make pomai-cache better for everyone. If you have questions about where to start, open an issue and tag it `help wanted` or `discussion`.

If you'd like, you can also add a `CODE_OF_CONDUCT.md` and standard GitHub templates (.github/ISSUE_TEMPLATE and .github/PULL_REQUEST_TEMPLATE) to further streamline contributions.

Happy hacking!
```