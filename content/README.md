# Homepage content guide

These files are the direct editing entry points for the live homepage. Changes take effect after commit without rebuilding the application.

- `publications.js`: publications
- `books.js`: books
- `patents.js`: patents
- `software-copyrights.js`: software copyrights
- `honors.js`: honors and awards
- `experience.js`: professional experience

To add a publication, copy one complete `{ ... },` entry in `publications.js`, edit its fields, and commit. Valid `group` values are `Journal Articles`, `Chinese-Language Journal Articles`, and `Conference Papers`.

For a new image, upload it to `images/` and set `image: "images/filename"`.

Do not edit `assets/index-*.js`; it is an automatically generated browser runtime file.
