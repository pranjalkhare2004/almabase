# ANTIGRAVITY_RULES.md

This file defines forbidden behaviors and anti-patterns.

------------------------------------------------------------------------

## 1. No Hallucination Zone

-   Never fabricate document names.
-   Never fabricate page numbers.
-   Never fabricate citations.
-   Never answer without retrieved evidence.

------------------------------------------------------------------------

## 2. No Monolithic Files

-   Do not create 1000-line files.
-   Do not centralize all logic into a single script.
-   Do not bypass service layers.

------------------------------------------------------------------------

## 3. No Over-Engineering

-   Avoid unnecessary abstraction.
-   Avoid premature optimization.
-   Avoid complex design patterns unless justified.

------------------------------------------------------------------------

## 4. No Security Negligence

-   Do not expose raw file paths.
-   Do not log secrets.
-   Do not hardcode API keys.

------------------------------------------------------------------------

## 5. No Silent Failures

-   All errors must be handled gracefully.
-   Log meaningful error messages.
-   Fail safe, not silently.

------------------------------------------------------------------------

## 6. No Data Corruption

-   Never modify original questionnaire text.
-   Never overwrite prior versions.
-   Always preserve structure integrity.

------------------------------------------------------------------------

## 7. No Prompt Drift

-   Do not change prompt templates without versioning.
-   Do not allow uncontrolled temperature changes.
-   Do not allow free-text AI answering without grounding.

------------------------------------------------------------------------

## 8. No Shortcuts

-   Do not skip authentication for convenience.
-   Do not skip validation.
-   Do not bypass citation enforcement.

------------------------------------------------------------------------

## 9. AI-Assisted Coding Guardrails

When generating code using AI: - Do not invent dependencies. - Do not
assume infrastructure. - Do not auto-generate unnecessary boilerplate. -
Prefer clarity over cleverness.

------------------------------------------------------------------------

## 10. Engineering Integrity Rule

If a feature compromises: - Grounding - Security - Structure - Data
integrity

It must not be implemented.
