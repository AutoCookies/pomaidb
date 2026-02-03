# Governance

## Maintainer roles
- **Maintainers**: Review and merge changes, approve releases, and resolve disputes.
- **Contributors**: Submit PRs and design proposals.

## Decision making
- Technical decisions are made via design docs or RFCs in `docs/`.
- Maintainers seek consensus; if blocked, a maintainer vote decides.

## Releases
- **Versioning**: Semantic Versioning (SemVer) for API and on-disk format changes.
- **Release process**:
  1. Cut a release branch.
  2. Run full test suite.
  3. Update release notes and compatibility notes.

## Breaking changes
- Breaking changes require:
  - A design doc/RFC.
  - Migration guidance in `docs/ON_DISK_FORMAT.md` (if storage-related).
  - A major version bump.

## Proposing design changes
- Create an RFC in `docs/` with:
  - Problem statement
  - Proposed design
  - Alternatives considered
  - Compatibility and migration plan
