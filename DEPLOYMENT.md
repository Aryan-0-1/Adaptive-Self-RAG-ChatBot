# Deploying to GitHub + Streamlit Community Cloud

A copy-pasteable walkthrough for taking this project from a local folder
to a live Streamlit Cloud app.

## 1. Initialize git and check `.gitignore` *before* your first commit

```bash
cd self-rag-assistant
git init
git add .gitignore
git commit -m "Add .gitignore"
```

Now confirm secrets won't be swept in:

```bash
git status
```

You should **not** see `.env`, `.streamlit/secrets.toml`, or any `*.sqlite3`
file listed as untracked-but-about-to-be-added. If you do, `.gitignore` isn't
catching them (double check the exact filename), fix it, then re-run
`git status` before proceeding.

```bash
git add .
git status   # sanity check again — no .env, no secrets.toml, no *.sqlite3
git commit -m "Initial commit"
```

**If you ever commit a real key by mistake:** rotate that key immediately
(assume it's compromised the moment it's pushed, even to a private repo),
then remove it from history:

```bash
# Using git-filter-repo (recommended, install via pip/brew):
pip install git-filter-repo
git filter-repo --path .env --invert-paths
git filter-repo --path .streamlit/secrets.toml --invert-paths

# Or using BFG Repo-Cleaner:
bfg --delete-files .env
bfg --delete-files secrets.toml
```

Then force-push the rewritten history (`git push --force`) — but rotating
the key is the part that actually matters; history rewriting is cleanup.

## 2. Create the GitHub repository and push

```bash
gh repo create self-rag-assistant --private --source=. --remote=origin
git push -u origin main
```

(Or create the repo in the GitHub UI first, then `git remote add origin
<url>` and `git push -u origin main`.)

## 3. Create the Streamlit Community Cloud app

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Pick your repository and branch (e.g. `main`).
4. Set **Main file path** to `frontend_self_rag.py`.
5. Click **Deploy**. It will fail on the first build until you add secrets
   (step 4 below) — that's expected.

## 4. Add secrets in the dashboard

1. Open `.streamlit/secrets.toml.example` in this repo and fill in real
   values for each key (your actual Google/Tavily/Ollama Cloud keys).
2. In the Streamlit Cloud dashboard, open your app → **Settings** →
   **Secrets**.
3. Paste the *filled-in* TOML content into the text box and **Save**.
4. The app will automatically restart. Whatever you pasted becomes
   available as `st.secrets["KEY_NAME"]` at runtime, which is exactly what
   `selfrag/config.py`'s `get_env()` checks first — no code changes needed
   between local and Cloud.

Never upload `secrets.toml` itself as a file to the repo — only paste its
contents into the dashboard box.

## 5. Confirm `requirements.txt` and Python version

Streamlit Cloud installs whatever's in `requirements.txt` at the repo root
verbatim — there's nothing else to configure for dependencies. If a
dependency requires a newer Python than Cloud's default:

1. In the app's **Settings** → **General**, there's a **Python version**
   dropdown — bump it to match what your dependencies need (check each
   package's own docs for its minimum supported Python version).
2. Save and let it redeploy.

## 6. Updating the deployed app / rolling back

- **Update:** push a new commit to the tracked branch — Streamlit Cloud
  auto-detects it and redeploys within a minute or two. No manual trigger
  needed.
- **Roll back:** either `git revert` the bad commit and push (preferred,
  keeps history honest), or push a hard reset (`git reset --hard <good-sha>`
  + `git push --force`) if you need to discard the bad commit entirely.
  Either way, once the branch Streamlit Cloud is watching points at the
  older commit, it redeploys automatically.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| App shows "GOOGLE_API_KEY isn't configured" banner | Secret key name in the dashboard doesn't exactly match `GOOGLE_API_KEY` (typo, wrong case, extra quotes) | Re-check the Secrets box against `.streamlit/secrets.toml.example` character-for-character |
| Build fails installing `faiss-cpu` | Some Cloud base images / Python versions don't have a prebuilt wheel for every `faiss-cpu` version | Pin `faiss-cpu` to a slightly older version known to have a manylinux wheel, or bump the Python version (step 5) and retry |
| Chat model calls hang or time out | You're using a **local** Ollama daemon locally, but Cloud has no way to reach `localhost:11434` on your machine | Set `OLLAMA_API_KEY` (Ollama Cloud) in Secrets — Cloud deployments can never reach a local daemon, only Ollama Cloud or another network-reachable endpoint |
| Web-search fallback never fires even for clearly web-answerable questions | `TAVILY_API_KEY` missing/misnamed in Secrets | Check the key name and that the "Web-Search Fallback" line under **🔧 Pipeline configuration** in the sidebar says "enabled" |
| Chat history / uploaded sources disappear after a while | Streamlit Cloud's filesystem is **ephemeral** — the SQLite file (`CHAT_DB_PATH`) is wiped on every redeploy/restart/sleep-wake cycle | Expected on Cloud with the current setup; wire `chat_store.py` to an external database (e.g. Postgres, Turso) if you need persistence across restarts |
