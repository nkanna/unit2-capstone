# Code Review Policy
**Document ID:** ENG-002 | **Version:** 1.7 | **Owner:** VP of Engineering

## Purpose
Consistent code review improves software quality, spreads knowledge across the team, and catches security issues before they reach production. This policy defines mandatory review requirements for all changes to Acme production codebases.

## Scope
Applies to all source code, infrastructure-as-code (Terraform, Helm charts), database migrations, and CI/CD pipeline definitions committed to any Acme-owned repository.

## Pull Request Size Guidelines
Pull requests should be small and focused to make reviews effective:
- **Preferred:** under 400 lines changed.
- **Acceptable:** 400–800 lines — requires a justification comment in the PR description.
- **Requires split or waiver:** over 800 lines — the author must break the change into smaller PRs or obtain written approval from their engineering manager before merging.
- Emergency hotfixes may bypass size limits but must be followed by a clean-up PR within five business days.

## Required Reviewers
- Every PR must have at least **two approvals** before merging to main or a release branch.
- At least one reviewer must be a domain owner (maintainer of the affected subsystem).
- PRs touching security-critical code (auth, payments, cryptography, secrets handling) require an additional approval from a member of the Security Review Board.
- The author may not approve their own PR.
- Reviews from automated bots do not count toward the human-reviewer minimum.

## Review Checklist
Reviewers are expected to verify the following:

**Correctness**
- Logic matches the stated intent of the ticket.
- Edge cases and error paths are handled.
- No off-by-one errors or unchecked null references.

**Security**
- No secrets, tokens, or credentials committed.
- User input is validated and sanitised before use.
- SQL queries use parameterised statements.
- No new dependencies added without a licence and vulnerability check.

**Testability and Coverage**
- Unit tests exist for new logic; existing tests are not deleted without justification.
- Happy paths and at least one failure path are covered.
- Tests do not depend on external services without mocking.

**Documentation and Style**
- Public APIs are documented (docstrings or OpenAPI specs).
- Code follows the project style guide (enforced by linter in CI).
- Commit messages follow the Conventional Commits format.

## Automated Checks (Must Pass Before Review)
All of the following CI checks must be green before a reviewer's time is requested:
- Unit test suite with minimum 80% line coverage.
- Static analysis (Ruff for Python, ESLint for TypeScript).
- Dependency vulnerability scan (Snyk or Dependabot alerts resolved).
- Secret scanning (detect-secrets pre-commit hook, Gitleaks in CI).

## Merge Requirements
- All requested changes are addressed or explicitly resolved.
- Branch is up-to-date with the target branch (no stale merges).
- All automated checks pass.
- PR has been open for a minimum of **two business hours** (prevents rubber-stamping).

## Review Turnaround SLA
- Authors should expect a first review within **one business day**.
- Reviewers must respond to re-review requests within **four business hours**.
- PRs open more than five business days without activity are auto-labelled `stale` and escalated to the engineering manager.

## Enforcement
Bypassing branch protections (e.g., force-pushing to main without review) requires prior approval from an engineering director and must be documented in the #incidents channel within one hour. Repeat violations without approval trigger a performance review.
