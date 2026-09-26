"""
Newsletter Draft Generator
---------------------------
Matches the client's (Finnish e-learning association) actual workflow:
  - They gather info from web pages / social posts about EVENTS and FIELD HIGHLIGHTS
  - They currently post interesting finds to a Telegram channel, then manually
    decide what to include in the newsletter
  - This script automates the "summarize" step: give it a list of links,
    it produces a short, original summary for each (not a copy of the source),
    plus the source link, organized by section (Events / Field Highlights)
  - Uses qwen2.5:7b via Ollama, running fully locally -> no per-request cost

Run locally (requires Ollama running: `ollama serve` in a separate terminal):
    python3 newsletter_generator.py
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from newspaper import Article
from ddgs import DDGS


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

# Kept for compatibility with earlier code that imports OLLAMA_URL directly.
OLLAMA_URL = OLLAMA_CHAT_URL

# --- Source reliability settings ---

# Sites that are either crowd-edited (Wikipedia), not primary sources, or
# generally not appropriate to cite in a professional newsletter.
# Add/remove domains here as your team decides what counts as reliable.
BLOCKED_DOMAINS = {
    "wikipedia.org",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "medium.com",  # often personal/unreviewed opinion pieces
}

# Domains/suffixes that get priority when present in search results -
# EU institutions, research bodies, and established European sources.
PREFERRED_DOMAIN_HINTS = [
    ".europa.eu", "cordis.europa.eu", "eurydice", "oecd.org",
    "elearningeuropa.info", ".ac.uk", ".edu",
    ".fi", ".de", ".fr", ".nl", ".se", ".dk", ".no",
]

# The client's own curated list of trusted sources they currently check
# manually (from sources.xlsx, provided directly by the client).
# name -> domain, used to populate a pick-list in the UI so the team can
# select sources instead of typing domains from memory.
CLIENT_TRUSTED_SOURCES = {
    "YLE (Finnish national broadcaster)": "yle.fi",
    "OPH - Finnish National Agency for Education": "oph.fi",
    "Ministry of Education and Culture (Finland)": "minedu.fi",
    "Ministry of Economic Affairs (Finland)": "tem.fi",
    "Theseus - AMK thesis repository": "theseus.fi",
    "Aalto University - News": "aalto.fi",
    "LUT University - News": "lut.fi",
    "HAMK Unlimited Journal": "unlimited.hamk.fi",
    "Aikuiskasvatus (adult education journal)": "aikuiskasvatus.fi",
    "Sitra (Finnish Innovation Fund)": "sitra.fi",
    "Karvi (Finnish Education Evaluation Centre)": "karvi.fi",
    "Association of Finnish Municipalities": "kuntaliitto.fi",
    "Finnish Institute of Occupational Health": "ttl.fi",
    "AOE - Finnish Open Educational Resources": "aoe.fi",
    "UNESCO Institute for Lifelong Learning": "uil.unesco.org",
    "UNICEF Education": "unicef.org",
    "Kopiosto (Finnish copyright org news)": "kopiosto.fi",
    "EDEN - European Distance & E-Learning Network": "eden-europe.eu",
    "ICDE - International Council for Open & Distance Education": "icde.org",
    "ALT - UK Association for Learning Technology": "alt.ac.uk",
    "Fleksibel Utdanning Norge (Norwegian, needs translation)": "fleksibelutdanning.no",
    "Gartner - Learning & Development": "gartner.com",
}

# File used to remember which URLs have already been used, so repeat
# searches for the same topic don't keep surfacing the same old article.
USED_LINKS_FILE = Path("used_links.json")


def ensure_scheme(url):
    """Add https:// to a URL if the person typed it without one
    (e.g. 'yle.fi' -> 'https://yle.fi'). Without this, requests/newspaper3k
    fail with a confusing 'no connection adapters' error."""
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


def _get_domain(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_blocked_domain(url):
    domain = _get_domain(url)
    return any(domain == blocked or domain.endswith("." + blocked)
           for blocked in BLOCKED_DOMAINS)


def is_preferred_domain(url):
    domain = _get_domain(url)
    return any(hint in domain for hint in PREFERRED_DOMAIN_HINTS)


def load_used_links():
    """Load the set of URLs already used in a previous newsletter."""
    if USED_LINKS_FILE.exists():
        try:
            return set(json.loads(USED_LINKS_FILE.read_text()))
        except Exception:
            return set()
    return set()


def mark_links_as_used(urls):
    """Add URLs to the persistent 'already used' memory."""
    used = load_used_links()
    used.update(urls)
    USED_LINKS_FILE.write_text(json.dumps(sorted(used), indent=2))


def clear_used_links():
    """Reset the duplicate-avoidance memory (e.g. for a fresh testing run)."""
    if USED_LINKS_FILE.exists():
        USED_LINKS_FILE.unlink()

# Anti-hallucination, copyright-safe instructions:
# - "in your own words" -> avoids copying source text directly (copyright)
# - "only use facts explicitly stated" -> reduces hallucination risk
# - short length -> keeps it a newsletter blurb, not a reproduction
# - explicit "in English" -> keeps the base summary language consistent
#   regardless of source article language, so translation to Finnish
#   (for the bilingual toggle) always translates FROM a known language.
SUMMARY_INSTRUCTION = (
    "Summarise this in 2-3 sentences, in English, in your own words, "
    "suitable for a newsletter blurb. Only use facts explicitly stated "
    "in the article above. Do not invent names, numbers, or details "
    "that are not in the text. Do not copy sentences directly from the "
    "article - rewrite in your own words."
)

TRANSLATE_INSTRUCTION = (
    "Translate the following newsletter title and summary into Finnish. "
    "Keep the meaning accurate and the tone natural for a newsletter. "
    "Respond in exactly this format, with nothing else before or after:\n"
    "TITLE_FI: <translated title>\n"
    "SUMMARY_FI: <translated summary>"
)

# Static translations for the section labels used in the app (Events /
# Field Highlights). If your team adds new section names, add their
# Finnish equivalent here too - otherwise the English name is reused.
SECTION_TITLE_FI = {
    "Events": "Tapahtumat",
    "Field Highlights": "Alan kohokohdat",
}


def get_article_text(url, include_image=False):
    """Download and extract the readable text of an article from a URL."""
    article = Article(url)
    article.download()
    article.parse()
    if include_image:
        return article.title, article.text, article.top_image or ""
    return article.title, article.text


def looks_like_homepage_or_section(url):
    """Heuristic: a URL with no real path (e.g. https://yle.fi or
    https://yle.fi/) is almost certainly a homepage/listing page,
    not a single article."""
    path = urlparse(url).path.strip("/")
    return path == ""


def discover_articles_from_source(homepage_url, limit=5):
    """
    Given a site's homepage/section URL (e.g. https://yle.fi), scan it for
    links to individual articles, using newspaper3k's built-in site-crawling
    (newspaper.build). Returns a list of article URLs found on that page.
    """
    from newspaper import build as build_source

    print(f"  {homepage_url} looks like a homepage - scanning for articles...")
    try:
        source = build_source(homepage_url, memoize_articles=False)
        urls = [a.url for a in source.articles]
        print(f"  Found {len(urls)} article links on the page.")
        return urls[:limit]
    except Exception as e:
        print(f"  Could not scan {homepage_url} for articles: {e}")
        return []


def expand_homepage_links(links, limit_per_site=5, avoid_repeats=True):
    """
    Given a mixed list of links, expand any homepage/section URLs into the
    individual article URLs found on them, so the client can paste a site
    like https://yle.fi directly instead of hunting for one specific
    article link themselves. Also applies the same blocklist + duplicate
    filtering used in topic search, for consistency.
    """
    used_links = load_used_links() if avoid_repeats else set()
    expanded = []

    for link in links:
        if looks_like_homepage_or_section(link):
            discovered = discover_articles_from_source(link, limit=limit_per_site)
            for url in discovered:
                if is_blocked_domain(url) or url in used_links:
                    continue
                expanded.append(url)
        else:
            expanded.append(link)

    return expanded


RELEVANCE_TOPICS_GUIDE = """
RELEVANT TOPICS INCLUDE (non-exhaustive):
- Education & learning: teaching, pedagogy, schools, universities, adult
  education, lifelong learning, curriculum, assessment, learning research
- Digital learning: e-learning, online/hybrid learning, LMS platforms,
  digital learning materials and methods
- Artificial intelligence: AI in education, AI tools, AI literacy, AI
  policy, responsible/ethical AI
- Educational technology: EdTech, learning analytics, VR/AR, adaptive
  and personalised learning
- Digital skills & competence: digital/media/information literacy,
  future skills, reskilling, upskilling
- Working life: future of work, hybrid work, workplace learning,
  digital transformation, automation
- Accessibility & responsibility: inclusive education, digital
  accessibility, data protection, sustainability
- Policy & development: education policy, EU digital education
  initiatives, research/funding projects
- Newsletter-worthy content: new reports, studies, funding
  opportunities, events, webinars, new tools/platforms, major policy
  changes, emerging trends

RELEVANCE SCALE:
1 = Completely unrelated (e.g. sports, celebrity news, unrelated politics)
2 = Weak connection - mentions education/tech/work but little useful content
3 = Potentially relevant - meaningful connection, a human should review it
4 = Clearly relevant - useful for professionals in digital learning/EdTech/AI
5 = Highly relevant - directly about digital learning, EdTech, AI in
    education, major research, significant policy or funding news

Do not score an article 1 just because it isn't specifically about
e-learning - education, teaching, skills, AI, and working-life changes
are all relevant. A clearly education/learning-related article should
normally score at least 3.
"""

ANALYSIS_INSTRUCTION = f"""
You are an information-monitoring assistant helping an e-learning
association decide what to include in their member newsletter.

Read the article below and return your analysis.

{RELEVANCE_TOPICS_GUIDE}

SUMMARY RULES (copyright-safe):
- Use ONLY facts explicitly stated in the article text provided.
- Do not invent names, numbers, or details not in the text.
- Do not copy sentences directly from the article - write the summary
  entirely in your own words (2-3 sentences, English, newsletter tone).

Return ONLY valid JSON, with exactly this structure and nothing else
before or after it:
{{
  "relevance": 3,
  "reason": "One short sentence explaining the relevance score.",
  "summary": "2-3 sentence original-wording summary in English.",
  "topics": ["topic 1", "topic 2"]
}}
"""


def analyze_and_summarise(title, text, source_label=""):
    """
    Analyze one article: score its relevance (1-5) for the newsletter,
    explain why, summarise it (copyright-safe, own words), and tag it
    with topics - all in a single structured call.

    Uses Ollama's /api/generate with format="json" and temperature=0,
    which is more reliable/parseable than free-form chat output.

    Returns a dict: {relevance, reason, summary, topics}. Falls back to
    a safe default (relevance=0) if the call or parsing fails, so a
    single bad article never crashes the whole batch.
    """
    max_chars = 6000
    trimmed_text = text[:max_chars]

    prompt = (
        f"{ANALYSIS_INSTRUCTION}\n\n"
        f"Source: {source_label}\n"
        f"Title: {title}\n\n"
        f"Article text:\n{trimmed_text}"
    )

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json()["response"]
        analysis = json.loads(raw)

        return {
            "relevance": int(analysis.get("relevance", 0) or 0),
            "reason": analysis.get("reason", ""),
            "summary": analysis.get("summary", ""),
            "topics": analysis.get("topics", []) or [],
        }

    except requests.exceptions.RequestException as e:
        print(f"  Could not reach Ollama for analysis: {e}")
        return {"relevance": 0, "reason": "Could not connect to Ollama.",
                "summary": "", "topics": []}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Could not parse AI analysis response: {e}")
        return {"relevance": 0, "reason": "The AI returned an unreadable response.",
                "summary": "", "topics": []}
    except Exception as e:
        print(f"  Unexpected error during analysis: {e}")
        return {"relevance": 0, "reason": "Unexpected error during analysis.",
                "summary": "", "topics": []}


def translate_to_finnish(title_en, summary_en):
    """
    Translate an English title + summary into Finnish, for the bilingual
    newsletter toggle. Returns (title_fi, summary_fi). Falls back to the
    English versions if translation fails or the response can't be parsed,
    so the newsletter still works even if this step has an issue.
    """
    prompt = f"Title: {title_en}\n\nSummary: {summary_en}\n\n{TRANSLATE_INSTRUCTION}"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"].strip()

        title_fi, summary_fi = title_en, summary_en
        for line in raw.splitlines():
            if line.strip().startswith("TITLE_FI:"):
                title_fi = line.split(":", 1)[1].strip()
            elif line.strip().startswith("SUMMARY_FI:"):
                summary_fi = line.split(":", 1)[1].strip()

        return title_fi, summary_fi

    except Exception as e:
        print(f"  Translation to Finnish failed, using English as fallback: {e}")
        return title_en, summary_en


def search_topic_for_links(topic, max_results=5, avoid_repeats=True, restrict_to_site=None):
    """
    Search the web for a topic and return a list of candidate URLs.
    Uses DuckDuckGo via the free `ddgs` library - no API key, no cost.

    Args:
        restrict_to_site: optional domain (e.g. "theseus.fi") to limit
            results to just that site - useful for trusted repositories
            you want to monitor specifically, rather than the open web.

    Applies three filters:
      1. Removes blocked/low-reliability domains (e.g. Wikipedia)
      2. Skips URLs already used in a previous newsletter (if avoid_repeats)
      3. Sorts remaining results so preferred EU/research domains come first

    Note: this only finds *links* (discovery). Downloading + summarising
    each one still uses get_article_text() and summarise_with_qwen() below,
    same as the manual-link workflow.
    """
    query = f"site:{restrict_to_site} {topic}" if restrict_to_site else topic
    print(f"Searching the web for: {query}")

    # Fetch extra results up front since some will get filtered out below.
    fetch_count = max_results * 4

    try:
        results = DDGS().text(query, max_results=fetch_count)
    except Exception as e:
        print(f"  Search failed: {e}")
        return []

    raw_urls = [r["href"] for r in results if r.get("href")]

    used_links = load_used_links() if avoid_repeats else set()

    filtered = []
    for url in raw_urls:
        if is_blocked_domain(url):
            continue
        if url in used_links:
            continue
        filtered.append(url)

    # Preferred (EU/research/edu) domains first, otherwise keep search order.
    filtered.sort(key=lambda u: 0 if is_preferred_domain(u) else 1)

    final_urls = filtered[:max_results]

    skipped_blocked = len(raw_urls) - len([u for u in raw_urls if not is_blocked_domain(u)])
    skipped_repeats = len([u for u in raw_urls if not is_blocked_domain(u) and u in used_links])

    print(f"  Found {len(raw_urls)} raw results -> "
          f"{skipped_blocked} blocked, {skipped_repeats} already used, "
          f"{len(final_urls)} kept.")

    return final_urls


def search_multiple_sites_for_links(topic, sites, max_results_total=5, avoid_repeats=True):
    """
    Search for a topic across several specific trusted sites at once
    (e.g. the client's curated source list), combining and deduplicating
    results. Each site already-used-link filtering applies per site, then
    results are combined and trimmed to max_results_total overall.

    Args:
        sites: list of domains, e.g. ["theseus.fi", "oph.fi"]
    """
    if not sites:
        return search_topic_for_links(topic, max_results=max_results_total, avoid_repeats=avoid_repeats)

    # Split the total budget roughly evenly across the selected sites.
    per_site = max(1, -(-max_results_total // len(sites)))  # ceil division

    combined = []
    seen = set()
    for site in sites:
        site_links = search_topic_for_links(
            topic, max_results=per_site, avoid_repeats=avoid_repeats, restrict_to_site=site
        )
        for url in site_links:
            if url not in seen:
                combined.append(url)
                seen.add(url)

    return combined[:max_results_total]


def build_newsletter_section_from_topic(section_title, topic, max_results=5,
                                         restrict_to_site=None, sites=None):
    """Search for a topic, then build a newsletter section from the results
    (combines discovery + summarisation in one step).

    Provide either `restrict_to_site` (a single domain) or `sites`
    (a list of domains to search across, e.g. the client's trusted list).
    """
    if sites:
        links = search_multiple_sites_for_links(topic, sites, max_results_total=max_results)
    else:
        links = search_topic_for_links(topic, max_results=max_results, restrict_to_site=restrict_to_site)

    return build_newsletter_section(section_title, links)


# --- RSS feed discovery ---
# RSS is a more reliable discovery method than scanning a homepage's raw
# HTML (see the YLE homepage-scanning issue) for any site that publishes
# a feed. Several of the client's trusted sources support this.
RSS_FEEDS = {
    "Opetushallitus (OPH)": "https://oph.fi/fi/latest.rss",
    "Theseus": "https://www.theseus.fi/feed/rss_2.0/site",
}


def fetch_rss_entries(feed_urls, limit_per_feed=10):
    """
    Fetch entries from one or more RSS feeds using feedparser.
    Returns a list of dicts: {title, link, description, source}.
    Applies the same blocklist + duplicate-avoidance used elsewhere.
    """
    import feedparser

    used_links = load_used_links()
    results = []

    for feed_url in feed_urls:
        print(f"Reading RSS feed: {feed_url}")
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"  Could not read feed {feed_url}: {e}")
            continue

        if getattr(feed, "bozo", False):
            print(f"  Feed warning for {feed_url}: {feed.bozo_exception}")

        for entry in feed.entries[:limit_per_feed]:
            link = entry.get("link", "")
            if not link or is_blocked_domain(link) or link in used_links:
                continue

            description = entry.get("summary", "") or entry.get("description", "")

            results.append({
                "title": entry.get("title", "No title"),
                "link": link,
                "description": description,
                "source": _get_domain(link),
            })

    return results


def build_newsletter_section_from_rss(section_title, feed_urls, limit_per_feed=10):
    """
    Fetch articles from RSS feeds, then analyse/summarise each one - same
    pipeline as the other discovery methods, so results are consistent
    (relevance scoring, bilingual translation at final build, etc.).
    """
    rss_entries = fetch_rss_entries(feed_urls, limit_per_feed=limit_per_feed)

    if not rss_entries:
        return {"section_title": section_title, "entries": []}

    print(f"\n--- Processing section: {section_title} (from RSS) ---")
    entries = []

    for rss_entry in rss_entries:
        url = rss_entry["link"]
        try:
            print(f"Downloading: {url}")
            # Try to get the full article text for a better summary;
            # fall back to the RSS description if scraping fails (e.g.
            # the site blocks scrapers or is JS-heavy).
            try:
                title, text, image_url = get_article_text(url, include_image=True)
                if not text or len(text) < 200:
                    raise ValueError("too little text extracted")
            except Exception:
                image_url = ""
                title = rss_entry["title"]
                text = rss_entry["description"]

            if not text or len(text) < 50:
                print(f"  Skipped (no usable content, even from RSS description): {url}")
                continue

            print("  Analysing and summarising...")
            analysis = analyze_and_summarise(title, text, source_label=rss_entry["source"])

            if not analysis["summary"]:
                print(f"  Skipped (AI analysis failed): {url}")
                continue

            entries.append({
                "title": title,
                "summary": analysis["summary"],
                "source_url": url,
                "image_url": image_url,
                "image_approved": False,
                "relevance": analysis["relevance"],
                "reason": analysis["reason"],
                "topics": analysis["topics"],
            })
            print(f"  Done. Relevance: {analysis['relevance']}/5")

        except Exception as e:
            print(f"  Failed to process {url}: {e}")

    return {"section_title": section_title, "entries": entries}


def build_newsletter_section(section_title, links):
    """Process a list of links into one newsletter section (e.g. 'Events').

    Any link that looks like a homepage/section page (e.g. https://yle.fi)
    is automatically expanded into the individual article links found on
    that page - so pasting a site's homepage works, not just direct
    article URLs.
    """
    print(f"\n--- Processing section: {section_title} ---")

    links = [ensure_scheme(link) for link in links if link.strip()]
    links = expand_homepage_links(links)
    entries = []

    for url in links:
        try:
            print(f"Downloading: {url}")
            title, text, image_url = get_article_text(url, include_image=True)

            if not text or len(text) < 200:
                print(f"  Skipped (too little text extracted): {url}")
                continue

            print("  Analysing and summarising...")
            analysis = analyze_and_summarise(title, text, source_label=_get_domain(url))

            if not analysis["summary"]:
                print(f"  Skipped (AI analysis failed or returned no summary): {url}")
                continue

            entries.append({
                "title": title,
                "summary": analysis["summary"],
                "source_url": url,
                "image_url": image_url,
                "image_approved": False,
                "relevance": analysis["relevance"],
                "reason": analysis["reason"],
                "topics": analysis["topics"],
            })
            print(f"  Done. Relevance: {analysis['relevance']}/5")

        except Exception as e:
            print(f"  Failed to process {url}: {e}")

    return {"section_title": section_title, "entries": entries}


def ask_ai_about_entries(question, all_entries):
    """
    Answer a free-form question about the currently gathered entries
    (e.g. "what trends do you see?"). Keeps everything grounded in the
    actual gathered data rather than letting the model invent things.
    """
    if not all_entries:
        context = "No articles have been gathered yet."
    else:
        context_parts = []
        for i, entry in enumerate(all_entries, start=1):
            topics = ", ".join(entry.get("topics", []) or [])
            context_parts.append(
                f"Article {i}\n"
                f"Title: {entry.get('title', '')}\n"
                f"Relevance: {entry.get('relevance', '-')}/5\n"
                f"Summary: {entry.get('summary', '')}\n"
                f"Topics: {topics}\n"
                f"Source: {entry.get('source_url', '')}"
            )
        context = "\n\n".join(context_parts)

    prompt = (
        "You are an assistant helping a newsletter editor understand the "
        "articles they've gathered so far. Answer using ONLY the "
        "information below - do not invent articles or facts that "
        "aren't there. If you can't answer from this data, say so.\n\n"
        f"GATHERED ARTICLES:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except Exception as e:
        return f"Could not get an answer from the AI: {e}"


def format_newsletter_markdown(sections):
    """Turn the processed sections into a readable draft, ready for human review."""
    lines = ["# Newsletter Draft (AI-generated - please review before sending)\n"]

    for section in sections:
        lines.append(f"## {section['section_title']}\n")
        if not section["entries"]:
            lines.append("_No items processed for this section._\n")
            continue

        for entry in section["entries"]:
            lines.append(f"### {entry['title']}")
            lines.append(entry["summary"])
            lines.append(f"[Read more]({entry['source_url']})\n")

    return "\n".join(lines)


from newsletter_design import format_newsletter_html


if __name__ == "__main__":
    # --- EDIT THESE: replace with real links your team gathers each cycle ---
    events_links = [
        # "https://example.com/some-upcoming-event",
    ]

    field_highlight_links = [
        "https://www.hamk.fi/julkaisut/opiskeluhyvinvointi-ammattikorkeakoulussa-opiskelijoiden-kokemuksia-voimavaroista-ja-vaatimuksista/",
    ]
    # --------------------------------------------------------------------

    sections = [
        build_newsletter_section("Events", events_links),
        build_newsletter_section("Field Highlights", field_highlight_links),
    ]

    draft = format_newsletter_markdown(sections)

    print("\n\n=== FULL NEWSLETTER DRAFT ===\n")
    print(draft)

    with open("newsletter_draft.md", "w") as f:
        f.write(draft)
    print("\nSaved to newsletter_draft.md - open it, review, and edit before sending.")

    html_draft = format_newsletter_html(sections)
    with open("newsletter_draft.html", "w") as f:
        f.write(html_draft)
    print("Saved to newsletter_draft.html - open this one in a browser to see the visual version.")
