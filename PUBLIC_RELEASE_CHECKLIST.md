# Public release checklist

The current tracked tree is designed to contain source, documentation, tests,
reproducible summaries, and selected plots. Local secrets, raw profiler
captures, crash dumps, virtual environments, transient benchmark runs, Memray
heaps, generated CocoaPods backups, and raw logs are ignored.

Before changing GitHub visibility:

1. Run `python tools/audit_public_release.py` and the complete test suite.
2. Review `git status --ignored` and confirm no ignored file is intentionally
   required by a clean clone.
3. Scan every reachable commit with a dedicated history scanner such as
   `gitleaks git .` or GitHub secret scanning.
4. Rewrite history to remove old absolute home paths, raw profiler binaries,
   generated dependency backups, logs, and any commit-author email that should
   be replaced by a GitHub `noreply` address. This is a force-push operation and
   must be done only from a verified backup after all collaborators stop work.
5. Re-clone the rewritten repository into an empty directory, install it, run
   tests and tutorials, and verify all README links.
6. Enable GitHub secret scanning, push protection, Dependabot, branch
   protection, and private-vulnerability reporting before making it public.

`.gitignore` protects future commits; it does not erase content already stored
in Git history. The repository should remain private until step 4 is completed
and independently checked.
