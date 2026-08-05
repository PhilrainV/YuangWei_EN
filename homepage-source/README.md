# Homepage source

The live content is stored in the repository root `content/` directory. The build workflow copies those files into this Vite project before building.

Application structure:

- `src/App.tsx`: page components and interaction logic
- `src/content.ts`: content type definitions and runtime loader
- `src/styles.css`: page styling
- `../content/`: profile, education, publications, books, patents, software copyrights, honors, and experience

Most routine updates require editing only `../content/`.
