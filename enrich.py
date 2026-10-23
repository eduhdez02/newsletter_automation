# enrich.py
"""
Enriquecimiento de medios para los items finales (los ~15-18 que sí van a
salir en el newsletter). No se corre sobre el pool completo para no golpear
decenas de sitios en cada ejecución.

Para cada artículo:
- og:image (imagen real de la noticia, para las tarjetas del newsletter).
  Si no existe, se genera una imagen de respaldo con IA (image_gen.py) —
  nunca se deja de tener alguna imagen relevante por categoría.
- og:video / iframe de YouTube embebido en el propio artículo (solo si la
  fuente ya lo trae — no se busca video por keyword en YouTube)
- Texto completo del artículo vía readability-lxml, como mejor insumo para
  la redacción de Opus que el resumen truncado del RSS.
"""
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import image_gen

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 DAUCH-Newsletter-Bot"
    )
}
TIMEOUT = 10

_YOUTUBE_RE = re.compile(r"(youtube\.com/embed/|youtu\.be/|youtube\.com/watch)", re.I)


def _meta(soup: BeautifulSoup, *names: str) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find(
            "meta", attrs={"name": name}
        )
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def _find_youtube_embed(soup: BeautifulSoup) -> str | None:
    for iframe in soup.find_all("iframe", src=True):
        if _YOUTUBE_RE.search(iframe["src"]):
            return iframe["src"]
    return None


def _is_google_news_redirect(url: str) -> bool:
    """Los links de Google News RSS no son la URL real del artículo: son
    una página intermedia que requiere JavaScript para redirigir. Al
    raspar esa página, su og:image casi siempre es el ícono genérico de
    la app de Google News (no una foto de la noticia) — así que no se
    puede confiar en ninguna imagen sacada de ahí."""
    return urlparse(url).netloc == "news.google.com"


def _extract_article_text(html: str, url: str) -> str:
    try:
        from readability import Document

        doc = Document(html)
        text = BeautifulSoup(doc.summary(), "lxml").get_text(separator=" ", strip=True)
        return text[:6000]
    except Exception:
        return ""


def enrich_item(item: dict, date_str: str) -> dict:
    """Devuelve una copia del item con image_url / video_url / full_text si
    se pudieron obtener. Nunca lanza excepción — un sitio caído no debe
    tumbar la corrida completa."""
    enriched = dict(item)
    enriched.setdefault("image_url", None)
    enriched.setdefault("video_url", None)
    enriched.setdefault("full_text", "")
    enriched.setdefault("image_credit_name", None)
    enriched.setdefault("image_credit_url", None)

    url = item.get("link")
    if url:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            html = resp.text
            soup = BeautifulSoup(html, "lxml")

            if not _is_google_news_redirect(url):
                image_url = _meta(soup, "og:image", "twitter:image")
                if image_url:
                    enriched["image_url"] = image_url

            video_url = _meta(soup, "og:video", "og:video:url") or _find_youtube_embed(soup)
            if video_url:
                enriched["video_url"] = video_url

            enriched["full_text"] = _extract_article_text(html, url)
        except Exception as e:
            print(f"  ⚠️  No se pudo enriquecer {url}: {e}")

    if not enriched["image_url"]:
        fallback = image_gen.generate_fallback_image(enriched, date_str)
        if fallback:
            enriched["image_url"] = fallback["image_url"]
            enriched["image_credit_name"] = fallback["credit_name"]
            enriched["image_credit_url"] = fallback["credit_url"]

    return enriched


def enrich_items(items: list[dict], date_str: str) -> list[dict]:
    return [enrich_item(it, date_str) for it in items]
