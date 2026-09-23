# fetch_news.py
"""
Descarga y limpia las noticias de todas las fuentes definidas en sources.py
(feeds directos curados + búsquedas de Google News RSS).

Se descarga cada feed con `requests` (headers de navegador real, sigue
redirects) y el contenido ya bajado se le pasa a feedparser.parse(). Dejar
que feedparser abra la URL directamente (su comportamiento por default)
producía falsos negativos: varios sitios devuelven 404/redirect a HTML
para un User-Agent genérico, y feedparser no distingue eso de "no hay
feed" — simplemente reporta 0 entradas.
"""
import feedparser
import requests
from bs4 import BeautifulSoup

import config  # noqa: F401  (fuerza stdout/stderr a UTF-8 en Windows)
from sources import all_feeds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 DAUCH-Newsletter-Bot"
    )
}
TIMEOUT = 12


def clean_html(html: str) -> str:
    """Limpia HTML y regresa solo texto."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ", strip=True)


def _fetch_feed(feed_url: str):
    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  No se pudo descargar {feed_url}: {e}")
        return None
    return feedparser.parse(resp.content)


def fetch_news() -> list[dict]:
    """Descarga y limpia noticias de todas las fuentes configuradas."""
    all_news = []

    for feed_url in all_feeds():
        feed = _fetch_feed(feed_url)
        if feed is None:
            continue

        if not feed.entries:
            print(f"  ⚠️  Feed sin entradas: {feed_url}")
            continue

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "") or entry.get("description", "")
            link = entry.get("link", "")

            if not title or not link:
                continue

            all_news.append({
                "title": title,
                "summary": clean_html(summary),
                "link": link,
                "published": entry.get("published", "") or entry.get("updated", ""),
                "source_feed": feed_url,
            })

    return all_news


if __name__ == "__main__":
    news = fetch_news()
    print(f"Total noticias descargadas: {len(news)}")
    for n in news[:10]:
        print("-", n["title"])
