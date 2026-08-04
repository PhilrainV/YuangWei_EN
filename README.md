# Yuang Wei — Academic Homepage

Live site: <https://philrainv.github.io/YuangWei_EN/>

## Updating academic outputs

Edit the small files in `content/`:

- `content/publications.js`: publications
- `content/books.js`: books
- `content/patents.js`: patents
- `content/software-copyrights.js`: software copyrights
- `content/honors.js`: honors and awards
- `content/experience.js`: professional experience

Commit the change to `master`; the live homepage reads these files directly and does not require a rebuild. See [content/README.md](content/README.md) for field descriptions and an example.

For a new publication image, upload it to `images/` and add `image: "images/filename"` to the publication entry.

The browser runtime in `assets/` is split into page logic, React, icons, and world-map data. These generated files normally do not need manual editing.

Google Scholar metrics are refreshed by `.github/workflows/google_scholar_crawler.yaml`. The visitor map exposes only anonymized visitor IDs and approximate regions.
