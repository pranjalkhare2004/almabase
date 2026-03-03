# Almabase GTM Engineering Internship Assignment

## Simplified Implementation Guide for AI-Assisted Development

This document defines the exact scope and constraints for building the
assignment project.

IMPORTANT: This project does NOT require production-grade architecture.
This project does NOT require complex design patterns. This project does
NOT require microservices, queues, or advanced scaling.

The only goal: Build a clean, working, end-to-end system that can be
deployed and demonstrated.

------------------------------------------------------------------------

# 1. Objective

Build a Structured Questionnaire Answering Tool that:

1.  Allows user authentication
2.  Accepts questionnaire upload
3.  Accepts reference document upload
4.  Generates answers using AI
5.  Shows citations from reference documents
6.  Allows editing before export
7.  Exports a structured document

Keep it simple. Working \> Perfect.

------------------------------------------------------------------------

# 2. Scope Constraints (Very Important)

The AI tool MUST:

-   Use a simple backend (FastAPI / Flask / Node Express)
-   Use SQLite or PostgreSQL (simple schema)
-   Use a simple vector store (FAISS or in-memory)
-   Avoid complex repository patterns
-   Avoid over-abstraction
-   Avoid multi-layer enterprise architecture
-   Avoid unnecessary classes
-   Avoid over-engineering

This is an internship assignment, not a production SaaS build.

------------------------------------------------------------------------

# 3. Functional Requirements

## 3.1 Authentication

-   Simple email + password login
-   JWT-based session
-   No OAuth required

## 3.2 Questionnaire Upload

-   Accept PDF or text file
-   Extract plain text questions
-   Split by newline or numbering pattern
-   Preserve question order

## 3.3 Reference Documents

-   Accept text or PDF
-   Convert to plain text
-   Chunk into small pieces
-   Generate embeddings
-   Store in vector index

## 3.4 Answer Generation

For each question:

1.  Retrieve top relevant chunks
2.  Pass chunks + question to LLM
3.  Force LLM to answer ONLY using provided context
4.  If no strong match → return: "Not found in references."

Each answer must include at least one citation.

Simple citation format: \[DocumentName - Chunk X\]

------------------------------------------------------------------------

# 4. Review & Edit

-   Display: Question Generated Answer Citation

-   Allow user to edit answer before export

-   Save edited answer

No versioning required unless simple to implement.

------------------------------------------------------------------------

# 5. Export

-   Generate a simple PDF or DOCX
-   Maintain original question text
-   Insert answer directly below each question
-   Include citation under each answer

Formatting can be minimal.

------------------------------------------------------------------------

# 6. Recommended Simple Stack

Backend: - FastAPI

Database: - SQLite

Vector Store: - FAISS

Frontend: - Minimal HTML templates OR simple React

Deployment: - Render / Railway / Vercel (backend + frontend)

Keep deployment straightforward.

------------------------------------------------------------------------

# 7. AI Prompt Rules

Use a strict prompt:

"You must answer ONLY using the provided context. If the answer is not
present in the context, respond exactly with: 'Not found in
references.'"

Keep temperature low (0.1 -- 0.2).

------------------------------------------------------------------------

# 8. What NOT To Do

-   Do NOT implement microservices
-   Do NOT build complex architecture
-   Do NOT over-optimize
-   Do NOT implement background job queues
-   Do NOT build admin dashboards
-   Do NOT build multi-tenant systems

Keep code readable and simple.

------------------------------------------------------------------------

# 9. What Matters Most

-   Clear workflow
-   Correct grounding
-   Citations present
-   Clean UI to review answers
-   Working deployed link

That is sufficient for submission.

------------------------------------------------------------------------

# 10. Submission Expectation

Deliver:

-   Working deployed link
-   GitHub repository
-   Short README explaining:
    -   Industry chosen
    -   Fictional company
    -   Assumptions
    -   Trade-offs
    -   Improvements with more time

No need for enterprise-level code.

------------------------------------------------------------------------

END OF SIMPLIFIED PROJECT GUIDE
