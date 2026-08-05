# Homepage content guide

These files are the direct editing entry points for the live homepage. Changes take effect after commit without rebuilding the application.

- `publications.js`: publications
- `profile.js`: name, role, affiliation, portrait, email, bio, research interests, and academic links
- `education.js`: education, institution links, logos, degrees, supervisors, and locations
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

## Editing education

All education entries are centralized in `education.js`. Each `{ ... },` item represents one institution or period of study:

- `period`: start and end dates
- `institution` and `institutionUrl`: institution name and website
- `program`: school, field, and degree description
- `logo` and `logoAlt`: logo path and alternative text
- `supervisor` and `supervisorUrl`: supervisor name and profile; remove both if not applicable
- `location`: city and country/region

To add an entry, copy one complete item and edit its values. Upload a new logo to `images/` before setting its path.
