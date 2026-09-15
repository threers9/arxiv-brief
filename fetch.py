import feedparser
import json
import os
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

Path("data").mkdir(exist_ok=True)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"

def call_groq(prompt):
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        },
        timeout=30,
    )
    if r.status_code == 429:
        wait = float(r.headers.get("Retry-After", "5"))
        raise RuntimeError(f"rate limited, retry after {wait}s")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

KEYWORDS = json.loads(Path("config/keywords.json").read_text())["keywords"]

# Fetch today's hep-th announcement (new + cross-listed) from the arXiv RSS feed.
# The export.arxiv.org API now rate-limits (HTTP 429) shared runner IPs, and
# feedparser silently turns that into an empty feed, so fetch with requests and
# fail loudly on HTTP errors.
url = "https://rss.arxiv.org/atom/hep-th"
for attempt in range(5):
    r = requests.get(url, headers={"User-Agent": "arxiv-brief (github.com/threers9/arxiv-brief)"}, timeout=60)
    if r.status_code == 429 or r.status_code >= 500:
        wait = 30 * (attempt + 1)
        print(f"arXiv feed returned {r.status_code}, retrying in {wait}s")
        time.sleep(wait)
        continue
    break
r.raise_for_status()
feed = feedparser.parse(r.content)
if feed.bozo and not feed.entries:
    raise RuntimeError(f"Could not parse arXiv feed: {feed.bozo_exception}")

# Load IDs from recent previous briefs to skip duplicates
seen_ids = set()
prev_briefs = sorted(Path("data").glob("2*.json"), reverse=True)
for f in prev_briefs[:3]:  # check last 3 days of briefs
    try:
        prev = json.loads(f.read_text())
        seen_ids.update(p["arxiv_id"] for p in prev.get("papers", []))
    except Exception:
        pass
papers = []
for entry in feed.entries:
    # Skip replacements; keep new submissions and cross-lists
    if entry.get("arxiv_announce_type") not in ("new", "cross"):
        continue
    arxiv_id = entry.id.split(":")[-1]  # "oai:arXiv.org:2609.13163v1" -> "2609.13163v1"
    if arxiv_id in seen_ids:
        continue
    categories = [t["term"] for t in entry.tags]
    primary = categories[0] if categories else "unknown"
    # RSS summary is "arXiv:XXXX Announce Type: new \nAbstract: ..."
    abstract = entry.summary.split("Abstract:", 1)[-1]
    papers.append({
        "arxiv_id": arxiv_id,
        "title": entry.title.strip().replace("\n", " "),
        "authors": ", ".join(a.name for a in entry.get("authors", [])),
        "abstract": abstract.strip().replace("\n", " "),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "primary_category": primary,
        "categories": categories,
        "published": entry.published,
    })

print(f"Found {len(papers)} recent papers")

# Score and summarize each paper
kw_list = ", ".join(KEYWORDS)
for p in papers:
    prompt = f"""You are helping a theoretical physicist triage arXiv hep-th papers.

Paper title: {p['title']}
Abstract: {p['abstract']}

The physicist's interests are: {kw_list}

Respond ONLY with valid JSON in this exact format:
{{"summary": "two-sentence summary for a physicist", "relevance_score": <integer 0-10>, "matched_interests": [<list of matching interests from the list, or empty>]}}"""
    parsed = None
    for attempt in range(5):
        try:
            text = call_groq(prompt).strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            break
        except Exception as e:
            msg = str(e)
            if "retry after" in msg:
                wait = float(msg.split("retry after ")[1].rstrip("s"))
            else:
                wait = 2 ** attempt
            print(f"Attempt {attempt+1} failed for {p['arxiv_id']}: {e} — retrying in {wait}s")
            time.sleep(wait)
    if parsed:
        p["summary"] = parsed.get("summary", "")
        p["relevance_score"] = int(parsed.get("relevance_score", 0))
        p["matched_interests"] = parsed.get("matched_interests", [])
    else:
        p["summary"] = p["abstract"][:200] + "..."
        p["relevance_score"] = 0
        p["matched_interests"] = []
    time.sleep(2.5)

# Generate overall daily digest
digest = ""
if papers:
    # Sort by score so the digest emphasizes what's most relevant to you
    sorted_papers = sorted(papers, key=lambda x: -x["relevance_score"])
    paper_lines = "\n".join(
        f"- [{p['primary_category']}] {p['title']} — {p['summary']}"
        for p in sorted_papers
    )
    digest_prompt = f"""You are writing a daily digest of new hep-th arXiv papers for a theoretical physicist whose interests are: {kw_list}.

Below are today's papers with brief summaries. Write a 1-2 paragraph overview highlighting the main themes and any notable results, especially ones matching the physicist's interests. Do NOT list every paper — synthesize themes. Mention specific paper titles only when genuinely noteworthy.

Papers:
{paper_lines}

Respond ONLY with valid JSON: {{"digest": "your 1-2 paragraph overview here"}}"""
    print("Waiting 60s before digest to let rate limit reset...")
    time.sleep(60)
    for attempt in range(5):
        try:
            text = call_groq(digest_prompt).strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            digest = json.loads(text).get("digest", "")
            print("Generated daily digest")
            break
        except Exception as e:
            msg = str(e)
            if "retry after" in msg:
                wait = float(msg.split("retry after ")[1].rstrip("s"))
            else:
                wait = 2 ** attempt * 10
            print(f"Digest attempt {attempt+1} failed: {e} — retrying in {wait}s")
            time.sleep(wait)
    else:
        print("Digest generation failed after 5 attempts")
        digest = ""


# Report success/failure counts
succeeded = sum(1 for p in papers if p["relevance_score"] > 0 or p["matched_interests"])
failed = len(papers) - succeeded
if failed:
    print(f"WARNING: {failed}/{len(papers)} papers fell back to truncated abstract (LLM failed)")
else:
    print(f"All {len(papers)} papers scored successfully")

# Save today's brief
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
if papers:
    output = {"date": today, "digest": digest, "papers": papers, "keywords_used": KEYWORDS}
    Path(f"data/{today}.json").write_text(json.dumps(output, indent=2))
    Path("data/latest.json").write_text(json.dumps(output, indent=2))
    print(f"Saved {today}.json with {len(papers)} papers")
else:
    print(f"No new papers for {today} (weekend or holiday) — skipping write")

# Maintain an index of all available dates
data_files = sorted([f.stem for f in Path("data").glob("*.json") if f.stem not in ("latest", "index")], reverse=True)
Path("data/index.json").write_text(json.dumps({"dates": data_files}, indent=2))

