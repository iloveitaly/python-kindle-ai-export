# Python rebuild: implementation and discovery handoff

Build a Python CLI for exporting books from a user's Kindle web reader, with optional OpenAI or Gemini OCR and downstream exports. Use this repository as a source of behavioral requirements, fixtures, and failure cases. Validate browser behavior against the real reader before treating its current selectors or inferred metadata as reliable.

## 1. Start from the correct reference

The improved implementation is on `feature/reliable-export-gemini`. At the time of this handoff, its changes are **uncommitted**, including untracked source, tests, and documentation. Reading upstream GitHub or checking out only the branch's HEAD will miss them. Inspect the local working tree and `git status`, or obtain a snapshot that includes modified and untracked files. Preserve this reference while building the Python project separately.

Read these in order:

| Reference                                                                                                             | What it explains                                                                                  |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [Integration notes](integration.md)                                                                                   | Selected PRs, exact reviewed heads, attribution, and validation limits                            |
| [Types](../src/types.ts) and [data loaders](../src/book-data.ts)                                                      | Coordinate separation, versioning, compatibility, completeness checks                             |
| [Extraction tests](../tests/extraction.test.ts) and [simulated-reader tests](../tests/extraction-integration.test.ts) | Boundary evidence, multiple screens on one printed page, cancellation, restoration, late metadata |
| [Extractor](../src/extract-kindle-book.ts)                                                                            | Current browser workflow and network interception                                                 |
| [Transcription orchestration](../src/transcription.ts) and [providers](../src/providers/transcription.ts)             | Cache identities, publication protocol, bounded retries, response classification                  |
| [Chapter mapping](../src/book-chapters.ts)                                                                            | Preserving content when TOC information is incomplete                                             |
| [Export integration tests](../tests/exports-integration.test.ts)                                                      | Actual command execution and PDF/Markdown checks                                                  |
| [Environment example](../.env.example)                                                                                | Current configuration surface and independent OCR/TTS settings                                    |

The last validation of this working tree passed formatting, lint, type checking, and 44 regression tests. Text, image, and cover PDFs were rendered and visually inspected. **No live Amazon login, regional-account extraction, OpenAI/Gemini OCR, or TTS run was performed.** The simulated browser tests verify control flow under supplied assumptions; they do not prove that Amazon currently behaves that way.

The approved feature sources were PRs #40, #31, #36, #34, #23, #32, #9, #37, and #38, selectively adapted. PRs #21, #33, #35, #16, and #8 were excluded. Keep Ollama/Tesseract integration and broad dependency/toolchain changes out of the rebuild's initial scope. Gemini image submission was informed by the Lab-K9 fork, whose older SDK and retry behavior should not be copied directly. Source links and attribution are in the integration notes.

## 2. Preserve these product behaviors

The pipeline has independent stages:

1. Capture rendered screens and metadata from the reader.
2. Optionally transcribe those screens through the selected OCR provider.
3. Export complete captured images to PDF, or complete transcription to Markdown/text PDF/audio.

Image PDF export must work without an AI API key. Gemini is an optional **OCR** provider; selecting it does not select a narration provider. Keep OpenAI OCR as the compatibility default unless the user requests otherwise. Narration remains independently configured. The reference uses an external Calibre conversion for EPUB; it does not implement a native EPUB writer.

Capture all available book content, including front and back matter. Earlier code guessed where the main story ended from TOC labels such as “Acknowledgements” and “About the author.” That behavior was deliberately removed. Any user-requested excerpt or content filtering should be explicit and should retain its scope in the output manifest.

Keep these outcomes independent:

- **Capture coverage:** beginning/end confirmed, partial, or legacy unknown.
- **Transcription coverage:** every capture accounted for, or unfinished/failed.
- **Transcription fidelity:** whether the model accurately read the text; successful requests do not prove accuracy.
- **Chapter mapping:** exact, approximate, or unavailable. Preserve ordered content even without chapter boundaries.
- **Reading-position restoration:** succeeded, failed, or unavailable. A complete export can coexist with failed restoration.

Do not collapse these into one `success` flag.

## 3. Discover the reader before implementing the full pipeline

Use a small owned book for a bounded, visible-browser diagnostic run. Record browser/OS, region, edition identifiers when available, reader settings, and what was actually observed. Build sanitized fixtures from this run. Keep account cookies, credentials, signed URLs, reader tokens, and full private book content out of committed fixtures.

Answer the following questions with evidence:

| Question                                                                | Why it matters / required evidence                                                                                                                     |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Which regional reader host serves this account?                         | Navigation and response matching must use the same configured host. A `.com` check can silently discard `.ca` responses.                               |
| Are `startReading`, `YJmetadata.jsonp`, and `/renderer/render` visible? | Missing optional responses must not be confused with a missing book. Compare cold and warm profiles.                                                   |
| Does blocking service workers make relevant responses observable?       | Cached responses can evade ordinary interception. Confirm this against the chosen Python browser runtime.                                              |
| Where is the ASIN found?                                                | Some code paths look in query parameters, others in response fields. Validate case and presence; never associate another book's response by guesswork. |
| What is actually inside render TARs?                                    | Inspect `metadata.json`, `toc.json`, `location_map.json`, and any `page_data_*` files. Missing files are valid cases.                                  |
| What identifies a rendered screen and indicates it finished loading?    | A changed image URL is useful evidence but not necessarily a new screen. Re-rendering can change it without navigation.                                |
| How are beginning/end controls represented?                             | Disabled, hidden, or removed controls require different checks. A temporary loading state must not count as an endpoint.                               |
| Does “Go to Page” or “Go to Location” exist, and where does it land?    | A printed page may contain multiple screens. Navigation to that page often lands at its first screen.                                                  |
| What happens on a book with only locations?                             | Page-based stopping and chapter calculations do not apply.                                                                                             |
| Does the reader render correctly in the target execution environment?   | The original project reports cloud/VM rendering trouble attributed to WebGL; treat this as a hypothesis to verify, not a universal limitation.         |

Produce a short observation report before committing to selector and coordinate assumptions. Retain a small diagnostic command that can repeat these observations when Amazon changes its UI.

## 4. Model screens and coordinates correctly

A **capture index** is an ordinal in the extraction sequence. It is the ordering key, not a page identifier. For example:

| Capture index | Printed page | Roman page | Kindle location |
| ------------- | ------------ | ---------- | --------------- |
| 0             | absent       | i          | absent          |
| 1             | 1            | absent     | absent          |
| 2             | 1            | absent     | absent          |

All three captures are legitimate. Two screens sharing printed page 1 must not overwrite one another or be deduplicated. A footer such as “Location 140 of 2300” represents another coordinate system entirely. Roman page `i` and Arabic page `1` must also remain distinct.

Suggested records:

- A capture: ordered index, screenshot path/hash/dimensions, observed raw label, optional printed/Roman/location coordinates, optional verified internal position interval.
- A TOC entry: original title and nesting, optional position and coordinate mappings, and mapping provenance.
- A capture manifest: schema version, book/edition/region, render settings, captures, metadata provenance, boundary evidence, coverage and restoration outcomes.
- A transcription run: provider/model/options/prompt identity, manifest identity, per-capture outcomes, completion and output hashes.

Use nullable values for unknown coordinates. Some reference `Nav` fields retain `-1` sentinels and ambiguous inclusive/exclusive comments from older code; those are compatibility remnants, not a recommended Python schema. A declared `position` field does not mean the extractor populates it.

**Unresolved in this reference:** the relationship between `location_map.locations`, internal positions, and each rendered image was not established. Screenshot `position`/`endPosition` values are therefore not fabricated. A Python implementation must either verify those mappings or preserve the explicit absence of chapter mapping. A list index in a TAR file must not become a Kindle location by assumption.

Never use a last printed page number as a count of rendered screens. Never stop before saving the last screen. “Page N of N” can still have additional screens of text.

## 5. Browser lifecycle and extraction traps

Implement the reader driver as a state machine with testable observations and actions. Separate it from file storage and exports.

- Register network and blob interception before opening the reader. The current JS injection snapshots blob bytes when object URLs are created because the reader may revoke those URLs quickly. A later fetch of the URL can be too late.
- Keep the injected browser code small even in a Python project. The relevant objects live inside the browser; Python cannot directly inspect browser memory.
- Track asynchronous response-processing tasks explicitly. Awaiting page navigation does not imply TAR parsing has finished. Bound the drain, collect exceptions, and prevent late tasks from mutating a finalized manifest.
- Extract each TAR into a fresh directory. Reusing a directory can make an omitted file appear present because an older response left it behind. Validate archive paths and link handling before extracting network archives in Python.
- Bound retained blob bytes as well as the number of blobs. Rewinding can otherwise retain many pages before capture begins. Prefer original image resolution; a DOM screenshot is a fallback with potentially different dimensions.
- Wait for a loaded image and stable navigation state. Confirm the state did not change between observation and saving the screenshot. Record enough evidence to diagnose duplicates or skipped screens.
- A screenshot hash by itself cannot prove progress: identical blank or decorative screens can be legitimate. An image URL by itself cannot prove progress either. Test both scenarios using the live observations.
- Establish the beginning explicitly, then traverse forward. If rewind fails, mark coverage partial; do not silently label a mid-book run complete.
- Treat endpoint detection as a combination of stable loaded state and navigation evidence. The reference accepts repeatedly disabled controls, or stable missing controls after failed advancement with an enabled opposite control. This remains a heuristic to validate on the live reader.
- Distinguish a stalled middle screen from a confirmed endpoint. Use bounded waits, bounded attempts, and an overall capture limit. Report which bound was reached.
- Persistent browser state and account reading position are different. A fresh profile does not guarantee that the account opens at the beginning. Do not run multiple extractors against the same profile simultaneously.

Reading-position restoration deserves its own implementation and tests. The current reference records its target after changing layout settings, compares screenshot bytes after returning, and may advance within the same printed page to recover the exact screen. That is better than restoring only the page number, but it does not establish restoration of the user's pre-settings layout. Define what the Python tool restores, record the target before mutations where possible, and report limitations honestly.

On cancellation, stop capture but allow bounded cleanup/restoration to run. Do not reuse a cancellation condition that immediately aborts the restoration loop. Protect the final manifest write even when cleanup fails. Python cancellation and callback semantics need explicit handling; translating JS `finally` blocks mechanically is insufficient.

## 6. Metadata should enrich capture, not gate it

Use explicit user overrides first, followed by verified reader metadata, render metadata, regional product-page data, and an ASIN/unknown-author fallback. Preserve all authors. Name normalization depends on the source format; do not repeatedly reverse/split already normalized names.

Metadata may arrive after capture has started. Update initial fallbacks when later authoritative data arrives, while retaining user overrides. Record provenance so an inferred title is distinguishable from one returned by the reader.

Keep the product-page lookup and cover download optional. Missing title, author, cover, or TOC must not discard otherwise useful captured content. A cover URL is different from a local validated image path. Validate response status, size, and image decoding; download once when possible. Allow for different product-page markup by region.

Do not copy raw reader response objects wholesale into public output. The reference removes some sensitive fields, but its types and allowlists are not an exhaustive guarantee. Define the output fields deliberately.

## 7. OCR: completion, retry, and cache semantics

Give each provider one small image-to-result adapter. Share scheduling, timeouts, retries, caching, and manifest publication outside the adapters. Consult current official Python SDK documentation when implementing the adapters; TypeScript SDK exception objects and retry options are not Python API contracts.

The current behavior intentionally distinguishes:

- Successful text.
- Explicitly blank page: valid output with empty text.
- Empty response: absence of a usable answer, not proof of a blank page.
- Blocked/refused response.
- Truncated response.
- Transport/API failure.

Inspect provider completion metadata, not just whether a response contains text. Partial text returned with a token-limit finish reason must not be cached as complete. The current `<BLANK_PAGE>` sentinel is a convention rather than a formally unambiguous protocol; consider a structured result if supported, and test literal marker text as an edge case.

Use literal transcription instructions. Preserve original wording and paragraphs. Do not carry forward older prompts claiming that every book is evidence in a court case, or escalation intended to overcome refusals. Keep optional cleanup/normalization separate and retain raw results. OCR does not preserve embedded illustrations; image PDFs do.

Retry transient failures with capped attempts and backoff. Respect provider retry hints; if a server delay exceeds the tool's wait budget, stop/defer instead of retrying before that delay. Check SDK-internal retries so a six-attempt policy does not secretly become dozens of requests. A timed-out operation must actually be cancelled or accounted for; starting another request while the first continues can duplicate cost.

Authentication/model failures should stop new scheduling. Already in-flight requests may still finish. Limit concurrency according to provider/account capacity, with validated configuration values. Zero-temperature OCR is not a guarantee of deterministic or accurate text.

Cache successful OCR by image bytes plus provider, model, prompt version, and generation settings. Attach **current** capture coordinates when reusing text; do not carry old page metadata out of an OCR cache entry. Hashes establish identity, not OCR correctness. Decide whether model aliases are sufficient for reproducibility or whether a resolved model version should also be recorded.

The reference's publication sequence is intentional: write a `running` report, invalidate the old final transcript, checkpoint individual successes, then publish content and a completion report only when every expected capture is accounted for. A crash between those writes must fail closed. A `running` report may remain after interruption even though many cache entries are reusable.

Improve these reference limitations in the Python design:

- Add a per-book run lock or transactional run directory. Atomic file replacement is not a multi-process transaction. The TypeScript implementation does not prevent concurrent runs or duplicate requests for an identical uncached image.
- Prefer retaining the previous complete export under its run identity while a new run proceeds, rather than deleting the user's only final transcript. Never present the old result as the new run's result.
- Persist progress consistently and expose failed capture indices without logging book text or credentials.
- Use temporary files on the same filesystem for atomic replacement; define recovery for abandoned temporary files and interrupted publication.

## 8. Export without losing or inventing content

Build a shared chapter-assignment step for Markdown, text PDF, and narration. Preserve ordered capture content exactly once, including prefix and trailing content. Missing TOC entries and unmappable boundaries should degrade the structure, not the text coverage.

Prefer verified position intervals, then compatible observed coordinates. Do not compare page numbers with locations or select the “nearest” page without marking that mapping approximate. If stronger position evidence contradicts a weaker page match, reject the weaker match.

Two TOC entries can begin on the same screen. The reference combines their labels; it cannot split that screen's text at an exact intra-screen boundary. Preserve the original TOC and disclose this granularity in the Python model rather than claiming exact chapter segmentation. Unique Markdown anchors are required even when chapter titles repeat.

PDF modes should remain explicit:

- `images`: one rendered screen per PDF page, preserved proportions, no OCR dependency or duplicate prepended cover.
- `text`: requires complete transcription; use the optional cover or a title page.
- `auto`: use complete text when available; otherwise report the reason and use complete captured images. It must not silently hide partial capture.

Image-only PDFs have no OCR text layer. Their conversion to EPUB does not create a useful reflowable text book automatically. Decide separately whether searchable image PDFs or a native EPUB writer belong in later scope.

Test real exported files, not just in-memory structures. Inspect PDF page counts, first/final text, cropping, cover behavior, and visual layout. Font coverage is a separate requirement: Unicode-safe filenames do not imply that the reference's text PDF font can render every writing system.

Use filenames that are safe across target platforms, have bounded encoded length, and include a stable disambiguator such as ASIN. Support an explicit basename override. Treat the filename as presentation; book/run identity must not depend on a title staying unchanged.

Narration has independent credentials, model/voice settings, batching, and cache identity. Preview currently means the first captured screen, which may be a cover or copyright page. Document that behavior or explicitly introduce another preview policy. Split long paragraphs without losing words, respect the provider's actual length units, and test audio assembly separately from OCR. The existing audio path received less integration coverage than Markdown/PDF and should be verified before claiming parity.

## 9. Python architecture and migration details

A reasonable module boundary is `cli`, `reader`, `capture`, `metadata`, `models`, `storage`, `transcription/providers`, `chapters`, and `exports`. Keep parsing, chapter assignment, and retry classification pure where practical. Inject the browser, clock/sleep, provider client, and storage interfaces into workflows so tests can reproduce failures without live accounts.

Use explicit runtime validation and domain exceptions. The TypeScript `assert()` helper always executes; Python's `assert` can be disabled and must not enforce input validity, coverage, or publication requirements.

Avoid constructors/imports that start work. The reference scripts run `main()` at module import; use explicit Python entry points so imports remain safe for tests and library consumers. Ensure background task exceptions are observed, cancellation is propagated deliberately, and resources close through structured cleanup.

Decide the compatibility promise before importing existing output:

- Can the Python tool read the current TypeScript `metadata.json` and `content.json`?
- Will it import OCR cache entries, or only completed content and screenshots?
- Are copied outputs relocatable? Current screenshot paths commonly resolve relative to the repository working directory. Prefer paths relative to the manifest, with a documented legacy resolution rule.
- How will legacy coverage be represented? Absence of a version/completion field must remain unknown, even if a compatibility mode permits exporting it.

**Cross-language hash trap:** the reference hashes `JSON.stringify(...)` output. Python's ordinary JSON output can differ in spaces, property order, Unicode escaping, float representation, and missing-versus-null fields. Parsing and reserializing is not enough to reproduce those hashes. Either implement and fixture-test the exact legacy canonicalization, verify legacy output in a dedicated importer, or write a new versioned canonical encoding and regenerate manifests/caches explicitly. Do not silently accept hash mismatches or rewrite legacy evidence as verified.

Use a new schema version for incompatible serialization or status changes. Import must preserve original files, source schema, and provenance.

## 10. Suggested build order and acceptance gates

1. **Snapshot and observe.** Preserve this working tree; run its tests; collect the bounded live-reader observations above. Deliver an assumptions/evidence list and sanitized fixture set.
2. **Define contracts and storage.** Implement versioned models, relative paths, atomic publication, locking, explicit coverage, and compatibility decisions. Test crash points and legacy imports before browser automation.
3. **Build capture plus image PDF.** Prove beginning-to-end capture, repeated printed pages, location-only books, cancellation, and restoration without introducing OCR cost. Validate an image PDF visually.
4. **Add provider-neutral OCR.** Implement one provider, checkpoints, bounded retries, and all result states. Add the second provider using the same contract. Verify each with a small live sample before a full book.
5. **Add shared chapter mapping and text exports.** Prove every expected capture is retained exactly once, including front/final content; exercise absent/ambiguous TOCs and mismatched coordinates.
6. **Add filenames, covers, and narration.** Keep these independent from capture/OCR correctness. Validate a small audio preview and its assembled output. Keep external EPUB conversion documented unless native EPUB is explicitly added.
7. **Perform acceptance and document limits.** Run the matrix below and report observed coverage. Keep live gaps explicit rather than promoting mocked results to live success.

Minimum regression matrix:

| Area               | Cases that must be exercised                                                                                                                                                              |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reader coordinates | Arabic pages; Roman front matter; location-only footer; multiple screens per printed page; temporarily missing footer                                                                     |
| Boundaries         | Start mid-book; confirmed first/last; enabled-but-stuck next control; controls hidden during loading; identical-looking successive screens; image re-render without progress              |
| Browser lifecycle  | Cold/warm profile; regional host; late metadata; missing optional TAR files; changed selectors; cancellation; restoration failure; account position retained across profiles              |
| Storage            | Interrupted metadata/content publish; malformed cache; changed image; changed coordinates; concurrent runs; moved output directory; legacy serialization/hash compatibility               |
| OCR                | Text; explicit blank; empty response; blocked; truncated; timeout; 429 with retry hint; transient server/network errors; invalid credentials/model; resume after interruption             |
| Chapters           | No TOC; duplicate title; same-screen boundaries; nesting; unmapped prefix/tail; final chapter; conflicting coordinate evidence                                                            |
| Exports            | Image PDF with no key; text rejects partial; auto fallback is visible; missing/bad cover; non-ASCII filenames; required font coverage; final text present; audio batch/preview boundaries |

The final handoff should include reproducible commands, the exact schemas, fixtures, a validation report separating local and live evidence, and a concise list of remaining assumptions. Successful compilation and matching the TypeScript output are useful checks, but neither establishes that the book was captured completely or transcribed accurately.
