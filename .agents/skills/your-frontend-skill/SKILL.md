# Frontend Skill

Use this skill when changing `frontend/`.

## Rules

- Keep RAG, parsing, embeddings, vector stores, model calls, and secrets out of the browser.
- Call the backend through `NEXT_PUBLIC_API_BASE_URL`.
- Prefer small React components, typed API contracts, and explicit loading/error states.
- Keep the first screen as the working application, not a landing page.
- Verify responsive behavior at mobile and desktop widths.
- Run `npm run lint` and `npm run build` after frontend changes.

## Visual Direction

The product is an academic RAG workbench. Use a quiet, dense interface with clear document, chat, and source regions. Avoid decorative marketing layouts.
