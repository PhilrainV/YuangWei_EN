# Homepage content guide

These files are the direct editing entry points for the live homepage. Changes take effect after commit without rebuilding the application.

- `publications.js`: publications
- `profile.js`: name, role, affiliation, portrait, email, bio, research interests, and academic links
- `books.js`: books
- `patents.js`: patents
- `software-copyrights.js`: software copyrights
- `honors.js`: honors and awards
- `experience.js`: professional experience

To add a publication, copy one complete `{ ... },` entry in `publications.js`, edit its fields, and commit. Valid `group` values are `Journal Articles`, `Chinese-Language Journal Articles`, and `Conference Papers`.

For a new image, upload it to `images/` and set `image: "images/filename"`.

Do not edit `assets/index-*.js`; it is an automatically generated browser runtime file.

## Editing the profile

All personal details are centralized in `profile.js`:

- `name` and `chineseName`: English and Chinese names
- `title` and `affiliation`: role and institution
- `affiliationUrl`: institution website
- `avatar` and `avatarAlt`: portrait path and alternative text
- `location` and `email`: location and email address
- `bio`: one biography paragraph per array item
- `researchInterests`: research-area labels
- `links`: ResearchGate, GitHub, Google Scholar, and ORCID URLs

Edit the values and commit the file. Keep the field names and outer braces intact.
