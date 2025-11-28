import os
import feedparser
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

# Carga lista de fuentes RSS desde el .env (separadas por coma)
RSS_FEEDS = os.getenv("RSS_FEEDS", "").split(",")

def clean_html(html):
    """Limpia HTML y regresa solo texto."""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text()

def fetch_news():
    """Descarga y limpia noticias de las fuentes definidas."""
    all_news = []

    for feed_url in RSS_FEEDS:
        feed_url = feed_url.strip()
        if not feed_url:
            continue

        print(f"Descargando noticias de: {feed_url}")
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")

            clean_summary = clean_html(summary)

            all_news.append({
                "title": title,
                "summary": clean_summary,
                "link": link
            })

    return all_news

# Prueba local (solo si ejecutas directamente este archivo)
if __name__ == "__main__":
    news = fetch_news()
    print(f"Total noticias descargadas: {len(news)}")
    for n in news[:5]:
        print(n["title"])