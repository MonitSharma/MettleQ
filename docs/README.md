# MettleQ documentation site

This branch contains the static GitHub Pages site for MettleQ. It is deliberately
dependency-free: GitHub Pages can serve `docs/index.html` without a JavaScript
toolchain, while the repository remains the canonical source for notebooks,
technical details, raw benchmark evidence, and API code. It is based on the
`development` evidence branch so the selected frozen plots are versioned with
the tables that describe them.

## Deploying this branch

1. Push `codex/github-pages-docs-development` to GitHub.
2. In **Settings → Pages**, choose **GitHub Actions** as the source.
3. The included `pages.yml` workflow uploads the `docs/` directory.
4. The site will normally be available at
   `https://<owner>.github.io/<repository>/` after the first successful run.

If Pages is instead configured for a branch and folder, select this branch and
the `/docs` folder. The existing site remains visible while this branch is
being reviewed if Pages currently publishes from `main`; this branch does not
replace that source until the Pages setting is changed or the branch is merged.

Benchmark figures on the landing page are intentionally labeled with their
hardware and contract. Add new results only with the machine manifest, exact
workload, dependency versions, warm-up/repeat protocol, and accuracy gate.
