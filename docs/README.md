# Project Website

Static HTML, CSS and JavaScript. Open `index.html` or serve this directory:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

The page includes the abstract, framework, method comparison, scalability demonstrations and an anonymous citation. Video paths are configured in `assets/js/config.js`.

## Review Version

The page omits author names, affiliation text and email addresses, and uses an anonymous citation. The Paper button serves the supplied `SSR-MARL-v2.pdf`; the Code button links to the public repository. The university crest is retained as the favicon.

Publish this directory's contents. Set the Code link in `assets/js/config.js` and its HTML fallback in `index.html`.

## Files

- `index.html`: page content and tables.
- `assets/css/style.css`: responsive layout.
- `assets/js/`: video loading, tabs, figure preview and citation copying.
- `assets/images/`: framework, configurations and university favicon.
- `assets/videos/`: demonstration videos and provenance records.
- `assets/citation.bib`: anonymous citation.
- `assets/paper/SSR-MARL-v2.pdf`: supplied paper version without author information.

