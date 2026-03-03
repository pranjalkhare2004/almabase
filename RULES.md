# RULES.md

## 1. Architecture Principles

-   Follow strict layered architecture:
    -   API Layer → HTTP handling only
    -   Service Layer → Business logic
    -   Repository Layer → Database access
    -   AI Layer → RAG and LLM interactions
    -   Models/Schemas → Validation only
-   Never mix layers.
-   No LLM calls inside API routes.
-   No raw SQL inside services.

------------------------------------------------------------------------

## 2. RAG Enforcement Policy

-   All answers must be grounded in retrieved context.
-   Minimum one citation per answer.
-   If similarity threshold not met → return: "Not found in references."
-   No free-form hallucinated answers.

------------------------------------------------------------------------

## 3. Prompt Governance

-   All prompts stored in /app/prompts/
-   Prompts must be versioned.
-   No inline prompt strings inside business logic.
-   Deterministic generation:
    -   Temperature ≤ 0.2
    -   Top_p ≤ 0.9

------------------------------------------------------------------------

## 4. Code Quality Standards

-   Mandatory type hints.
-   Pydantic validation for all request/response schemas.
-   No file \> 300 lines.
-   No function \> 40 lines.
-   Clear naming conventions:
    -   \*\_service.py
    -   \*\_repository.py
    -   \*\_schema.py

------------------------------------------------------------------------

## 5. Security Rules

-   JWT authentication required.
-   Password hashing using bcrypt.
-   Validate file type and size.
-   No secrets in code.
-   Use .env with .env.example provided.

------------------------------------------------------------------------

## 6. Database Rules

-   Use migrations.
-   Enforce foreign keys.
-   Vector column indexed.
-   Store chunk metadata (document name, page, chunk_id).

------------------------------------------------------------------------

## 7. Logging & Observability

Log: - Retrieval scores - LLM latency - Token usage - User ID

Never log: - Passwords - Raw tokens - Sensitive document contents

------------------------------------------------------------------------

## 8. Testing Requirements

-   Unit tests for retrieval logic.
-   Unit tests for citation mapping.
-   Integration test for end-to-end workflow.
-   Mock LLM in tests.

------------------------------------------------------------------------

## 9. Export Integrity

-   Preserve original question text.
-   Maintain order.
-   Insert answers below questions.
-   Include citations inline.
-   Include confidence score (if implemented).

------------------------------------------------------------------------

## 10. Versioning Policy

-   Each generation creates a new version record.
-   User edits must not overwrite AI output.
-   Maintain comparison capability between versions.
