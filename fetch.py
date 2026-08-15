import feedparser
import google.generativeai as genai
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

Path("data").mkdir(exist_ok=True)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

KEYWORDS = json.loads(Path("config/keywords.json").read_text())["keywords"]

# Fetch recent hep-th papers (primary + cross-listed)
url = (
    "http://export.arxiv.org/api/query?"
    "search_query=cat:hep-th"
    "&sortBy=submittedDate&sortOrder=descending&max_results=150"
)
feed = feedparser.parse(url)

# Keep only papers announced in the last 24h
cutoff = datetime.now(timezone.utc) - timedelta(hours=36)
papers = []
for entry in feed.entries:
    published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if published < cutoff:
        continue
    arxiv_id = entry.id.split("/abs/")[-1]
    categories = [t["term"] for t in entry.tags]
    primary = categories[0] if categories else "unknown"
    papers.append({
        "arxiv_id": arxiv_id,
        "title": entry.title.strip().replace("\n", " "),
        "authors": ", ".join(a.name for a in entry.authors),
        "abstract": entry.summary.strip().replace("\n", " "),
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
    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
        p["summary"] = parsed.get("summary", "")
        p["relevance_score"] = int(parsed.get("relevance_score", 0))
        p["matched_interests"] = parsed.get("matched_interests", [])
    except Exception as e:
        print(f"Error on {p['arxiv_id']}: {e}")
        p["summary"] = p["abstract"][:200] + "..."
        p["relevance_score"] = 0
        p["matched_interests"] = []
    time.sleep(0.5)  # Stay under Gemini rate limit

# Save today's brief
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
if papers:
    output = {"date": today, "papers": papers, "keywords_used": KEYWORDS}
    Path(f"data/{today}.json").write_text(json.dumps(output, indent=2))
    Path("data/latest.json").write_text(json.dumps(output, indent=2))
    print(f"Saved {today}.json with {len(papers)} papers")
else:
    print(f"No new papers for {today} (weekend or holiday) — skipping write")
    
# Maintain an index of all available dates
data_files = sorted([f.stem for f in Path("data").glob("*.json") if f.stem != "latest"], reverse=True)
Path("data/index.json").write_text(json.dumps({"dates": data_files}, indent=2))

print(f"Saved {today}.json with {len(papers)} papers")