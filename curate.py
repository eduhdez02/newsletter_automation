# curate.py
"""
Pipeline de curación: dedup -> filtro duro -> clasificación con Claude Haiku
(relevancia + categoría vía structured output) -> selección con cuota por
categoría.

Reemplaza la fórmula ad-hoc de scoring del summarizer.py original (que
pesaba 0.5 a la recencia y dejaba pasar noticias irrelevantes solo por ser
recientes) por un criterio editorial real, barato de correr en volumen
porque usa Claude Haiku 4.5 en lotes.
"""
import unicodedata
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse

from anthropic import Anthropic
from pydantic import BaseModel

import config
from sources import CATEGORIES, NEGATIVE_KEYWORDS, POSITIVE_OVERRIDE_KEYWORDS

try:
    from dateutil import parser as dateparser
except ImportError:
    dateparser = None

BATCH_SIZE = 20

# Newsletter quincenal: cualquier cosa más vieja que esto ya no es
# "novedad". Filtrar por recencia antes de clasificar reduce muchísimo el
# volumen que llega al LLM (Google News RSS regresa resultados históricos
# para una búsqueda, no solo lo reciente) sin costo de una sola llamada.
MAX_AGE_DAYS = 30

# Cuántos items finales queremos por categoría en el newsletter
CATEGORY_QUOTAS = {
    "competencia": 4,
    "equipos_geoespaciales": 4,
    "procesamiento_ia": 3,
    "construccion_innovacion": 3,
    "tendencias_regulacion": 3,
}

CategoryKey = Literal[
    "competencia",
    "equipos_geoespaciales",
    "procesamiento_ia",
    "construccion_innovacion",
    "tendencias_regulacion",
    "irrelevante",
]


class ClassifiedItem(BaseModel):
    index: int
    is_relevant: bool
    category: CategoryKey
    relevance_score: int
    reason: str


class ClassificationBatch(BaseModel):
    items: list[ClassifiedItem]


def _normalize_text(t: str) -> str:
    if not t:
        return ""
    t2 = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii", "ignore")
    return t2.lower().strip()


def _domain_from_link(link: str) -> str:
    try:
        return urlparse(link).netloc.replace("www.", "")
    except Exception:
        return ""


def _parse_date(item: dict):
    val = item.get("published")
    if not val:
        return None
    try:
        if dateparser:
            return dateparser.parse(str(val))
        return datetime.fromisoformat(str(val))
    except Exception:
        return None


def dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for it in items:
        key = _normalize_text(it.get("title", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def filter_by_recency(items: list[dict], max_age_days: int = MAX_AGE_DAYS) -> list[dict]:
    now = datetime.now(timezone.utc)
    out = []
    for it in items:
        dt = _parse_date(it)
        if not dt:
            # Sin fecha parseable: se conserva (mejor incluir de más que
            # perder una noticia por un feed con formato de fecha raro).
            out.append(it)
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if (now - dt).days <= max_age_days:
            out.append(it)
    return out


def hard_filter(items: list[dict]) -> list[dict]:
    """Descarta ruido evidente (militar, racing, consumo) antes de gastar
    una sola llamada al modelo."""
    out = []
    for it in items:
        text = _normalize_text(it.get("title", "") + " " + it.get("summary", ""))
        has_negative = any(kw in text for kw in NEGATIVE_KEYWORDS)
        has_positive = any(kw in text for kw in POSITIVE_OVERRIDE_KEYWORDS)
        if has_negative and not has_positive:
            continue
        out.append(it)
    return out


def _classify_batch(client: Anthropic, batch: list[dict]) -> list[ClassifiedItem]:
    categories_desc = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    lines = []
    for i, it in enumerate(batch):
        domain = _domain_from_link(it.get("link", ""))
        lines.append(
            f"[{i}] Título: {it.get('title', '')}\n"
            f"    Resumen: {(it.get('summary', '') or '')[:300]}\n"
            f"    Fuente: {domain}"
        )
    prompt = f"""Eres el editor de un newsletter B2B de DAUCH, una consultora de \
levantamientos geoespaciales con drones. Clasifica cada noticia numerada según \
estas categorías:

{categories_desc}
- irrelevante: no aplica a ninguna categoría anterior (drones militares, \
hobby/racing, fotografía de eventos, noticias genéricas de aviación sin \
relación con topografía/geoespacial/construcción/IA aplicada al sector).

Para cada noticia da: si es relevante (is_relevant), su categoría, un \
relevance_score de 0 a 10 (10 = altamente relevante y accionable para un \
lector técnico/comercial del sector), y una razón breve.

Noticias:
{chr(10).join(lines)}
"""
    response = client.messages.parse(
        model=config.CLASSIFIER_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
        output_format=ClassificationBatch,
    )
    return response.parsed_output.items


def classify_items(items: list[dict]) -> list[dict]:
    """Anota cada item con category / relevance_score y descarta los
    irrelevantes."""
    client = Anthropic(**config.anthropic_client_kwargs())
    classified = []

    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        try:
            results = _classify_batch(client, batch)
        except Exception as e:
            print(f"  ⚠️  Falló la clasificación de un lote: {e}")
            continue

        for r in results:
            if r.index < 0 or r.index >= len(batch):
                continue
            if not r.is_relevant or r.category == "irrelevante":
                continue
            item = dict(batch[r.index])
            item["category"] = r.category
            item["relevance_score"] = r.relevance_score
            classified.append(item)

    return classified


def select_final_items(items: list[dict]) -> list[dict]:
    """Aplica cuota por categoría, ordenando por relevance_score y
    recencia dentro de cada categoría."""
    now = datetime.now(timezone.utc)
    by_category: dict[str, list[dict]] = {k: [] for k in CATEGORY_QUOTAS}

    for it in items:
        cat = it.get("category")
        if cat in by_category:
            by_category[cat].append(it)

    def sort_key(it):
        dt = _parse_date(it)
        recency = 0.0
        if dt:
            try:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                recency = -(now - dt).total_seconds()
            except Exception:
                recency = 0.0
        return (it.get("relevance_score", 0), recency)

    final = []
    for cat, quota in CATEGORY_QUOTAS.items():
        bucket = sorted(by_category.get(cat, []), key=sort_key, reverse=True)
        final.extend(bucket[:quota])

    return final


def curate(raw_items: list[dict]) -> list[dict]:
    deduped = dedupe(raw_items)
    recent = filter_by_recency(deduped)
    filtered = hard_filter(recent)
    print(
        f"  -> {len(filtered)} noticias tras dedupe + recencia (<= {MAX_AGE_DAYS} días) "
        f"+ filtro duro (de {len(raw_items)} en bruto)."
    )

    classified = classify_items(filtered)
    print(f"  -> {len(classified)} noticias marcadas como relevantes por el clasificador.")

    final = select_final_items(classified)
    print(f"  -> {len(final)} noticias finales seleccionadas (con cuota por categoría).")
    return final
