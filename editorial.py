# editorial.py
"""
Redacción final del newsletter con Claude Opus 5, usando structured outputs
(Pydantic) en vez de la reparación de JSON por regex del summarizer.py
original.

Decisión de diseño importante: el modelo NUNCA es la fuente de verdad para
URLs (source_link, image_url, video_url). Esas vienen siempre de los datos
que ya enriquecimos en enrich.py; el modelo solo redacta título/resumen/
empresa. Esto evita el riesgo de que un LLM "invente" o corrompa un link.
"""
import re
from urllib.parse import urlparse

from anthropic import Anthropic
from pydantic import BaseModel

import config
from curate import CATEGORY_QUOTAS


class NewsItem(BaseModel):
    title: str
    summary: str
    company: str | None = None
    source_name: str
    source_link: str
    image_url: str | None = None
    video_url: str | None = None
    published: str | None = None


class Newsletter(BaseModel):
    edition_number: int
    date: str
    summary_exec: str
    competencia: list[NewsItem]
    equipos_geoespaciales: list[NewsItem]
    procesamiento_ia: list[NewsItem]
    construccion_innovacion: list[NewsItem]
    tendencias_regulacion: list[NewsItem]
    recommendation: str


class EditorialItem(BaseModel):
    index: int
    title: str
    summary: str
    company: str | None = None


class EditorialOutput(BaseModel):
    summary_exec: str
    items: list[EditorialItem]
    recommendation: str


def _domain_from_link(link: str) -> str:
    try:
        return urlparse(link).netloc.replace("www.", "")
    except Exception:
        return ""


def _pretty_source_name(domain: str) -> str:
    if not domain:
        return "Fuente"
    base = domain.split(".")[0]
    return base.replace("-", " ").title()


def is_summary_weak(text: str) -> bool:
    if not text:
        return True
    txt = text.strip()
    words = txt.split()
    return len(words) < 20 or len(txt) < 120


def _build_prompt(final_items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(final_items):
        body = it.get("full_text") or it.get("summary") or ""
        lines.append(
            f"[{i}] Categoría: {it.get('category')}\n"
            f"    Título original: {it.get('title', '')}\n"
            f"    Contenido: {body[:1200]}\n"
            f"    Fuente: {_domain_from_link(it.get('link', ''))}"
        )
    return f"""Eres el editor técnico del newsletter quincenal de DAUCH, una \
consultora de levantamientos geoespaciales con drones (topografía, LiDAR, \
fotogrametría, GNSS). El público lector son consultoras de ingeniería, \
constructoras y clientes técnicos que quieren estar al día del sector: \
competencia (DroneDeploy, Propeller Aero, Skycatch...), equipos \
geoespaciales nuevos, software de procesamiento con IA, innovación en \
construcción, y tendencias del mercado.

Para cada noticia numerada, redacta en ESPAÑOL:
- title: título claro y directo (puedes pulir el original)
- summary: 40-80 palabras explicando QUÉ pasó y POR QUÉ le importa a una \
consultora geoespacial mexicana. Sé específico, nunca genérico.
- company: la empresa protagonista si aplica (o null)

Además escribe:
- summary_exec: resumen ejecutivo de 3-4 líneas en texto plano: (1) qué \
está pasando en el sector en esta quincena, (2) por qué importa para DAUCH, \
(3) una lectura estratégica. Nada de relleno ni frases vacías tipo \
"innovaciones en el sector".
- recommendation: una recomendación concreta y accionable para DAUCH \
(1-2 frases).

Noticias:
{chr(10).join(lines)}
"""


def _regenerate_summary_exec(client: Anthropic, newsletter_items: list[NewsItem]) -> str:
    bullets = "\n".join(f"- {it.title}: {it.summary}" for it in newsletter_items[:10])
    prompt = f"""Escribe en ESPAÑOL un resumen ejecutivo de 3-4 líneas para un \
newsletter B2B de DAUCH (consultora geoespacial), a partir de estas noticias:

{bullets}

Debe cubrir: (1) qué está pasando en el sector, (2) por qué importa para \
DAUCH, (3) una lectura estratégica. Texto plano, sin JSON, sin viñetas, \
tono profesional y específico. Máximo 4 líneas."""
    response = client.messages.create(
        model=config.EDITOR_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip()


def build_newsletter(final_items: list[dict], edition_number: int, date_str: str) -> Newsletter:
    client = Anthropic(**config.anthropic_client_kwargs())
    prompt = _build_prompt(final_items)

    response = client.messages.parse(
        model=config.EDITOR_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
        output_format=EditorialOutput,
    )
    editorial = response.parsed_output

    by_index = {e.index: e for e in editorial.items}
    grouped: dict[str, list[NewsItem]] = {k: [] for k in CATEGORY_QUOTAS}

    for i, raw in enumerate(final_items):
        ed = by_index.get(i)
        if not ed:
            continue
        domain = _domain_from_link(raw.get("link", ""))
        news_item = NewsItem(
            title=ed.title or raw.get("title", ""),
            summary=ed.summary,
            company=ed.company,
            source_name=_pretty_source_name(domain),
            source_link=raw.get("link", ""),
            image_url=raw.get("image_url"),
            video_url=raw.get("video_url"),
            published=raw.get("published"),
        )
        cat = raw.get("category")
        if cat in grouped:
            grouped[cat].append(news_item)

    summary_exec = editorial.summary_exec
    if is_summary_weak(summary_exec):
        print("  ⚠️  Resumen ejecutivo débil, regenerando...")
        all_items = [it for items in grouped.values() for it in items]
        try:
            regenerated = _regenerate_summary_exec(client, all_items)
            if regenerated and not is_summary_weak(regenerated):
                summary_exec = regenerated
        except Exception as e:
            print(f"  ⚠️  No se pudo regenerar el resumen ejecutivo: {e}")

    return Newsletter(
        edition_number=edition_number,
        date=date_str,
        summary_exec=summary_exec,
        competencia=grouped.get("competencia", []),
        equipos_geoespaciales=grouped.get("equipos_geoespaciales", []),
        procesamiento_ia=grouped.get("procesamiento_ia", []),
        construccion_innovacion=grouped.get("construccion_innovacion", []),
        tendencias_regulacion=grouped.get("tendencias_regulacion", []),
        recommendation=editorial.recommendation,
    )
