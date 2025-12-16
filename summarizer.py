# summarizer.py
"""
Summarizer completo para DAUCH — Newsletter
- selección inteligente de noticias (dedupe, scoring, diversidad)
- meta-summarize (compresión por lotes) para poder procesar muchas noticias
- prompt estricto en español que devuelve JSON
- reparación automática del JSON si modelo devuelve texto mezclado
- copia automática del logo a output/newsletters/assets/logo.png
- render Jinja2 a output/newsletters/newsletter_YYYY-MM-DD.html

Usar desde la raíz del proyecto:
python summarizer.py
"""
import os
import json
import re
import shutil
import unicodedata
from math import exp
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

# tu módulo que obtiene noticias (fetch_news.py)
from fetch_news import fetch_news

load_dotenv()

# Configuración
# --------------------------------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError(
        "❌ OPENAI_API_KEY no está definida. "
        "Configúrala en .env (local) o en GitHub Secrets (Actions)."
    )

ROOT = Path(__file__).parent.resolve()
TEMPLATE_DIR = ROOT / "templates"
OUTPUT_DIR = ROOT / "output" / "newsletters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ruta alternativa del logo que me compartiste (se intentará usar)
UPLOADED_LOGO_PATH = Path("/mnt/data/ed9a5350-ef05-430e-a104-496b9852740f.png")

env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))

# -------------------------
# Selección inteligente de noticias (dedupe, scoring, diversidad)
# -------------------------
try:
    from dateutil import parser as dateparser
except Exception:
    dateparser = None

KEYWORDS = [
    "lidar", "photogrammetry", "fotogrametr", "gnss", "rtk", "gpr",
    "sensor", "sensors", "pix4d", "dji", "terra", "machine learning",
    "inteligencia artificial", "ai", "ml", "inspección", "topografía", "mapping",
    "ortofoto", "multiespectral", "hiperespectral", "thermal", "térmico", "termico",
    "drone", "drones", "autonomous", "automatización", "autónomo", "workflow", "pipeline",
    "zenmuse", "inspección"
]

SOURCE_PRIORITY = {
    "pix4d.com": 0.18,
    "lidarmag.com": 0.15,
    "dronedj.com": 0.12,
    "emlid.com": 0.10,
    "sph-engineering.com": 0.08,
    "arxiv.org": 0.20,
    "towardsdatascience.com": 0.10,
    "geospatialworld.net": 0.11
}

def _normalize_text(t: str) -> str:
    if not t:
        return ""
    t2 = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii", "ignore")
    return t2.lower().strip()

def _parse_date(item):
    # intentar parsear varios campos de fecha
    candidates = []
    if isinstance(item, dict):
        for key in ("published", "published_parsed", "pubDate", "updated", "date", "created"):
            val = item.get(key)
            if val:
                candidates.append(val)
    for val in candidates:
        try:
            if dateparser:
                return dateparser.parse(str(val))
            else:
                # intentar iso
                return datetime.fromisoformat(str(val))
        except Exception:
            continue
    return None

from urllib.parse import urlparse
def _domain_from_link(link):
    try:
        p = urlparse(link)
        return p.netloc.replace("www.", "")
    except Exception:
        return ""

def compute_score(item, now_dt):
    title = _normalize_text(item.get("title",""))
    summary = _normalize_text(item.get("summary","") or item.get("description","") or "")
    link = item.get("link","") or ""
    domain = _domain_from_link(link)

    # recency
    dt = _parse_date(item)
    if dt:
        delta_days = (now_dt - dt).days
        recency_score = max(0.0, exp(-delta_days / 10.0))
    else:
        recency_score = 0.08

    # keyword score
    kw_count = 0
    for kw in KEYWORDS:
        if kw.lower() in title or kw.lower() in summary:
            kw_count += 1
    keyword_score = min(1.0, kw_count * 0.35)

    # length score
    length = len(summary)
    length_score = min(1.0, length / 350.0)

    # source priority
    src_boost = SOURCE_PRIORITY.get(domain, 0.0)

    # combine
    score = (0.5 * recency_score) + (0.28 * keyword_score) + (0.12 * length_score) + src_boost
    return score

def select_top_items(items, k=10, max_per_source=2):
    """
    Dedup, score, enforce diversity. Devuelve up to k items.
    """
    now_dt = datetime.now(timezone.utc)
    seen_titles = set()
    uniq = []
    for it in items:
        t = _normalize_text(it.get("title",""))
        if not t:
            continue
        if t in seen_titles:
            continue
        seen_titles.add(t)
        uniq.append(it)

    scored = []
    for it in uniq:
        s = compute_score(it, now_dt)
        scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    source_count = {}
    for score, it in scored:
        if len(selected) >= k:
            break
        domain = _domain_from_link(it.get("link",""))
        cnt = source_count.get(domain, 0)
        if cnt >= max_per_source:
            continue
        selected.append(it)
        source_count[domain] = cnt + 1

    if len(selected) < k:
        for score, it in scored:
            if it in selected:
                continue
            selected.append(it)
            if len(selected) >= k:
                break

    return selected

# -------------------------
# Meta-summarize: agrupar muchos items y resumir por lote
# -------------------------
def chunk_list(lst, n):
    """Divide lista en chunks de tamaño n"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def meta_summarize_many(items, batch_size=8, model="gpt-4o-mini"):
    """
    Si items es grande, agrupa en batches y llama al modelo para resumir cada batch.
    Devuelve una lista de items 'comprimidos' con keys title, summary, source_name, source_link, published.
    """
    if not items:
        return []
    # si pocos items, devolver tal cual
    if len(items) <= batch_size:
        # normalize keys
        out = []
        for it in items:
            out.append({
                "title": it.get("title",""),
                "summary": it.get("summary","") or it.get("description","") or "",
                "source_name": it.get("source","") or "",
                "source_link": it.get("link","") or "",
                "published": it.get("published","") or it.get("date","") or ""
            })
        return out

    compressed = []
    # para cada chunk, pedimos al modelo un resumen condensado con JSON
    for idx, chunk in enumerate(chunk_list(items, batch_size), start=1):
        # construir mini-prompt con los items del chunk
        chunk_body = []
        for i, it in enumerate(chunk, start=1):
            t = it.get("title","").strip()
            s = it.get("summary","") or it.get("description","") or ""
            link = it.get("link","") or ""
            chunk_body.append(f"{i}) {t}\nExtracto: {s[:400]}...\nFuente: {link}\n")
        chunk_text = "\n".join(chunk_body)

        prompt = f"""
Eres un editor técnico. A partir de la siguiente lista de noticias, resume cada noticia en un objeto JSON con:
- title (titular)
- summary (2-3 frases concisas en ESPAÑOL, 30-80 palabras)
- source_name (nombre del medio si aparece) 
- source_link (URL)
- published (si existe, sino "")
Devuelve solo un JSON con la lista de objetos: [{{...}}, ...]. Nada fuera del JSON.

Noticias (batch {idx}):
{chunk_text}
"""
        try:
            # llamar al modelo (reusar call_openai_chat definido abajo) - usar temperatura baja
            text = call_openai_chat(prompt, model=model, temperature=0.05, max_tokens=600)
            parsed = extract_json_with_regex(text)
            if parsed and isinstance(parsed, list):
                # añadir cada objeto
                for o in parsed:
                    # asegurar campos mínimos
                    compressed.append({
                        "title": o.get("title",""),
                        "summary": o.get("summary",""),
                        "source_name": o.get("source_name",""),
                        "source_link": o.get("source_link",""),
                        "published": o.get("published","")
                    })
                continue
            # fallback simple: convertir chunk items to compressed basic
            for it in chunk:
                compressed.append({
                    "title": it.get("title",""),
                    "summary": (it.get("summary","") or it.get("description",""))[:180] + "...",
                    "source_name": "",
                    "source_link": it.get("link",""),
                    "published": it.get("published","") or ""
                })
        except Exception as e:
            # en error, fallback simple
            for it in chunk:
                compressed.append({
                    "title": it.get("title",""),
                    "summary": (it.get("summary","") or it.get("description",""))[:180] + "...",
                    "source_name": "",
                    "source_link": it.get("link",""),
                    "published": it.get("published","") or ""
                })
    # limitar resultados (por si se agruparon muchos)
    return compressed[:max(10, len(items)//2)]

# -------------------------
# Prompt builder (few-shot, estricto JSON, español)
# -------------------------
def build_prompt_from_items(items, max_items=8):
    """
    Construye prompt (few-shot) para generar el JSON final.
    items: lista de dict con title, summary, source_link (lo usamos como input)
    """
    parts = []
    for i, it in enumerate(items[:max_items], start=1):
        t = it.get("title", "").strip()
        s = it.get("summary", "").strip()
        link = it.get("source_link", "") or it.get("link","")
        s_short = (s[:600] + "...") if len(s) > 600 else s
        parts.append(f"{i}) {t}\nExtracto: {s_short}\nFuente: {link}\n")

    body = "\n".join(parts)

    example = {
      "title":"DAUCH — Newsletter",
      "date":"2025-11-01",
      "summary_exec":"Resumen ejecutivo de ejemplo en 3 líneas sobre impacto para consultoras.",
      "news":[
        {
          "title":"Ejemplo: Lanzamiento X",
          "summary":"Resumen claro (2–4 frases) que explique qué pasó y por qué importa para operaciones.",
          "source_name":"Medio Ejemplo",
          "source_link":"https://ejemplo.com/noticia",
          "published":"2025-11-01"
        }
      ],
      "tech":[
        {"text":"Ejemplo: Nuevo sensor LiDAR con mayor alcance.","source_name":"Medio Ejemplo","source_link":"https://ejemplo.com/tech"}
      ],
      "companies":[
        {"name":"Empresa Ejemplo","country":"País","service":"Mapeo","product_or_method":"Método X","note":"Por qué importa","link":"https://ejemplo.com/empresa"}
      ],
      "trends":[
        {"text":"Ejemplo: Regulación X afecta operaciones","source_name":"Organismo","source_link":"https://ejemplo.com/reg"}
      ],
      "recommendation":"Ej: Priorizar pruebas de campo con sensor X."
    }

    prompt = f"""
Eres un editor técnico profesional. A partir de las noticias listadas (máx {max_items}) GENERA SÓLO UN JSON VÁLIDO y en ESPAÑOL. NADA fuera del JSON.

IMPORTANTE: responde únicamente con JSON que siga exactamente la estructura del ejemplo. Usa este EJEMPLO como plantilla.

EJEMPLO_SALIDA:
{json.dumps(example, ensure_ascii=False, indent=2)}

AHORA LAS NOTICIAS (INPUT):
{body}

REGLAS:
- Devuelve solo JSON válido con las claves: title,date,summary_exec,news,tech,companies,trends,recommendation.
- news: 6–8 items (cada summary 40–120 palabras).
- tech y trends: objetos con text, source_name, source_link.
- companies: 4–6 empresas con product_or_method si la noticia lo menciona.
- No inventes datos. Si no hay información, usa "".
- Mantén longitud total < 15 minutos de lectura.

Devuelve únicamente el JSON en ESPAÑOL.
"""
    return prompt.strip()

# -------------------------
# OpenAI call y extracción robusta
# -------------------------
def call_openai_chat(prompt, model="gpt-4o-mini", temperature=0.12, max_tokens=1600):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada en .env")
    # import lazy
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)

    messages = [
        {"role": "system", "content": "Eres un editor técnico profesional. Responde en ESPAÑOL. Devuelve SOLO JSON válido según el esquema solicitado."},
        {"role": "user", "content": prompt}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # extracción robusta
    try:
        choice0 = resp.choices[0]
        msg = getattr(choice0, "message", None)
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list) and len(content) > 0:
                first = content[0]
                if isinstance(first, dict):
                    return first.get("text") or first.get("content") or json.dumps(first, ensure_ascii=False)
                return str(first)
            return str(content) if content is not None else str(msg)
        else:
            if hasattr(msg, "content"):
                cont = getattr(msg, "content")
                if isinstance(cont, (list, tuple)) and len(cont) > 0:
                    first = cont[0]
                    if isinstance(first, dict):
                        return first.get("text") or first.get("content") or str(first)
                    return str(first)
                return str(cont)
            if hasattr(msg, "text"):
                return str(getattr(msg, "text"))
            return str(msg)
    except Exception:
        if hasattr(resp, "text") and resp.text:
            return str(resp.text)
        return str(resp)

# -------------------------
# JSON helpers: parse + repair
# -------------------------
def parse_json_safe(text):
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            sub = text[start:end+1]
            try:
                return json.loads(sub)
            except Exception:
                pass
    return None

def extract_json_with_regex(text):
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end+1].strip()
        return json.loads(candidate)
    except Exception:
        pattern = re.compile(r"(\{(?:.|\n)*\})", re.MULTILINE)
        m = pattern.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None

def repair_json_with_model(generated_text):
    if not OPENAI_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        system = (
            "Eres un asistente que extrae JSON. Te daré un texto que contiene un JSON mezclado "
            "con explicaciones u otros datos. Devuélveme ÚNICAMENTE el JSON válido que coincide "
            "con las claves: title,date,summary_exec,news,tech,companies,trends,recommendation. "
            "Si no encuentras JSON válido, responde con {}."
        )
        user = "Texto a procesar (EXTRAE el JSON correcto y nada más):\n\n" + generated_text[:4000]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=1200
        )
        # extraer texto de la respuesta
        try:
            choice0 = resp.choices[0]
            msg = getattr(choice0, "message", None)
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    text = content[0] if len(content)>0 else ""
                else:
                    text = content or ""
            else:
                text = getattr(msg, "content", "") if hasattr(msg, "content") else getattr(msg, "text", "")
        except Exception:
            text = getattr(resp, "text", str(resp))
        parsed = extract_json_with_regex(text)
        return parsed
    except Exception as e:
        print("repair_json_with_model falló:", e)
        return None

def ensure_structured_json(generated_text, items):
    if not generated_text:
        return local_fallback_structured(items)
    parsed = parse_json_safe(generated_text)
    if parsed:
        return parsed
    parsed = extract_json_with_regex(generated_text)
    if parsed:
        return parsed
    print("Salida no JSON — intentando reparación automática con el modelo...")
    repaired = repair_json_with_model(generated_text)
    if repaired:
        return repaired
    print("No fue posible reparar: usando fallback local.")
    return local_fallback_structured(items)

# -------------------------
# Si summary_exec es débil, regenerarlo de forma dirigida
# -------------------------
def is_summary_weak(text):
    """
    Determina si el resumen ejecutivo es demasiado corto o genérico.
    Ajusta umbrales si quieres (min_words, min_chars).
    """
    if not text:
        return True
    txt = text.strip()
    # condiciones simples: menos de 20 palabras o demasiado genérico (frases muy cortas)
    words = txt.split()
    if len(words) < 20 or len(txt) < 120:
        return True
    # heurística: frases cortas repetitivas
    if txt.lower().startswith("resumen") or "innovación" in txt.lower() and len(words) < 30:
        return True
    return False

def regenerate_summary_exec(parsed, final_items, model="gpt-4o-mini"):
    """
    Genera un resumen ejecutivo en TEXTO plano (3-4 líneas) en ESPAÑOL.
    Si el modelo devuelve JSON, lo convierte a texto.
    """
    # Construir contexto breve: titulares + resúmenes de news (hasta 8)
    news_block = []
    for n in parsed.get("news", [])[:8]:
        title = n.get("title","").strip()
        summary = n.get("summary","").strip()
        if title and summary:
            news_block.append(f"- {title}: {summary}")
        elif title:
            news_block.append(f"- {title}")
    tech_block = []
    for t in parsed.get("tech", [])[:4]:
        if isinstance(t, dict):
            tech_block.append(f"- {t.get('text','')}")
        else:
            tech_block.append(f"- {t}")
    comp_block = []
    for c in parsed.get("companies", [])[:4]:
        name = c.get("name","").strip()
        prod = c.get("product_or_method","") or c.get("product","") or ""
        if name:
            comp_block.append(f"- {name}: {prod}")

    prompt = f"""
Eres un editor ejecutivo. A partir de los elementos abajo, escribe en ESPAÑOL un Resumen Ejecutivo en TEXTO PLANO de 3–4 líneas:
1) Qué ocurrió en general (una frase).
2) Por qué importa para una consultora geoespacial como DAUCH (una frase).
3) Una acción prioritaria recomendada (una frase).

Noticias (titular + extracto):
{chr(10).join(news_block)}

Tecnologías destacadas:
{chr(10).join(tech_block)}

Empresas/productos relevantes:
{chr(10).join(comp_block)}

INSTRUCCIONES IMPORTANTES:
- Devuelve SOLO texto plano (sin JSON, sin etiquetas).
- Máximo 4 líneas; cada línea 1–2 frases.
- Tono profesional, accionable y específico.
"""
    try:
        text = call_openai_chat(prompt, model=model, temperature=0.05, max_tokens=260)
        res = (text or "").strip()

        # Si el asistente devolvió JSON, parsearlo y convertirlo a texto legible
        if res.startswith("{") or res.startswith("["):
            try:
                parsed_obj = json.loads(res)
                # Si estructura conocida: buscar keys típicas
                if isinstance(parsed_obj, dict):
                    # Manejar varios esquemas posibles
                    if "resumen_ejecutivo" in parsed_obj:
                        re = parsed_obj["resumen_ejecutivo"]
                        # si viene como dict con campos
                        if isinstance(re, dict):
                            evento = re.get("evento","").strip()
                            importancia = re.get("importancia","").strip()
                            accion = re.get("accion_prioritaria","") or re.get("accion","") or ""
                            lines = []
                            if evento: lines.append(evento)
                            if importancia: lines.append(importancia)
                            if accion: lines.append(accion)
                            return "\n".join(lines)[:1000]
                    # si viene con keys directas
                    parts = []
                    for k in ("summary_exec","summary","resumen"):
                        if k in parsed_obj and isinstance(parsed_obj[k], str):
                            parts.append(parsed_obj[k].strip())
                    if parts:
                        return "\n".join(parts)[:1000]
                # si es lista, concatenar textos
                if isinstance(parsed_obj, list):
                    texts = []
                    for item in parsed_obj:
                        if isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, str):
                                    texts.append(v.strip())
                        elif isinstance(item, str):
                            texts.append(item.strip())
                    if texts:
                        return "\n".join(texts)[:1000]
            except Exception:
                # si no pudimos parsear JSON, limpiamos el texto bruto
                res = re.sub(r'[\{\}\[\]"\\]', '', res)
                res = re.sub(r'\s{2,}', ' ', res).strip()
                # cortar a 1000 chars para seguridad
                return res[:1000]

        # Si ya es texto plano, devolver tal cual (limpiando saltos dobles)
        res = re.sub(r'[\r\n]{3,}', '\n\n', res)
        # Asegurar máximo 4 líneas: tomar primeras 4 líneas no vacías
        lines = [ln.strip() for ln in res.splitlines() if ln.strip()]
        if len(lines) > 4:
            lines = lines[:4]
        return "\n".join(lines).strip()
    except Exception as e:
        print("No se pudo regenerar summary_exec con el modelo:", e)
        return parsed.get("summary_exec","")

# -------------------------
# Fallback local estructurado
# -------------------------
def local_fallback_structured(items, max_companies=4, max_tech=4, max_trends=3):
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary_exec = "Resumen ejecutivo (generado localmente): selección de novedades y fuentes relevantes."
    tech = []
    companies = []
    trends = []
    for it in items[:max_tech]:
        tech.append((it.get("title","")[:160]).strip())
    for it in items[:max_companies]:
        companies.append({
            "name": it.get("title","")[:80],
            "country": "",
            "service": "",
            "product_or_method": "",
            "note": "",
            "link": it.get("link","")
        })
    for it in items[:max_trends]:
        trends.append((it.get("title","")[:140]).strip())
    recommendation = "Recomendación: revisar las fuentes listadas y priorizar pruebas de campo en sensores o proveedores relevantes."
    return {
        "title":"DAUCH — Newsletter",
        "date": date,
        "summary_exec": summary_exec,
        "news": [{"title":it.get("title",""), "summary": (it.get("summary","") or it.get("description","") or "")[:250], "source_name":"", "source_link":it.get("link",""), "published":""} for it in items[:8]],
        "tech": tech,
        "companies": companies,
        "trends": trends,
        "recommendation": recommendation
    }

# -------------------------
# Copy assets (logo)
# -------------------------
def copy_assets():
    src1 = TEMPLATE_DIR / "assets" / "logo.png"
    dst_dir = OUTPUT_DIR / "assets"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "logo.png"
    if UPLOADED_LOGO_PATH.exists():
        try:
            shutil.copy2(UPLOADED_LOGO_PATH, dst)
            print("Logo copiado desde ruta subida a:", dst)
            return
        except Exception as e:
            print("No se pudo copiar logo desde ruta subida:", e)
    if src1.exists():
        try:
            shutil.copy2(src1, dst)
            print("Logo copiado desde templates/assets/logo.png a:", dst)
            return
        except Exception as e:
            print("No se pudo copiar el logo desde templates:", e)
    print("Advertencia: no se encontró logo en ninguna ruta. Coloca templates/assets/logo.png o la ruta subida.")

# -------------------------
# Render HTML
# -------------------------
def render_html_from_struct(data, output_path):
    tpl_name = "newsletter_template.html"
    tpl = env.get_template(tpl_name)
    summary_exec = data.get("summary_exec", "")
    news = data.get("news", []) or []
    tech = data.get("tech", []) or []
    companies = data.get("companies", []) or []
    trends = data.get("trends", []) or []
    recommendation = data.get("recommendation", "")
    # usar timezone.utc correctamente
    date = data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    html = tpl.render(
        date=date,
        summary_exec=summary_exec,
        news=news,
        tech=tech,
        companies=companies,
        trends=trends,
        recommendation=recommendation
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return output_path

# -------------------------
# Main flow
# -------------------------
def main():
    print("1) Obteniendo noticias...")
    items = fetch_news()
    if not items:
        print("No se encontraron noticias. Revisa RSS_FEEDS en .env")
        return
    print(f"  -> {len(items)} noticias obtenidas.")

    # Selección inteligente: dedupe, score y diversidad
    items_selected = select_top_items(items, k=20, max_per_source=2)  # seleccionar un pool mayor primero
    print(f"  -> Seleccionadas para compresión: {len(items_selected)} items (pool para meta-summarize).")

    # Meta-summarize: si el pool es grande, comprimimos por batches
    # Parámetros que puedes ajustar:
    batch_size = 6      # cuántos items por batch al comprimir
    compressed = meta_summarize_many(items_selected, batch_size=batch_size, model="gpt-4o-mini")
    print(f"  -> Comprimidos a {len(compressed)} items tras meta-summarize.")

    # Para el prompt final, seleccionamos top N de los comprimidos
    final_items = select_top_items(compressed, k=8, max_per_source=2)
    print(f"  -> Final items para prompt: {len(final_items)}")

    prompt = build_prompt_from_items(final_items, max_items=len(final_items))

    print("2) Llamando al modelo (OpenAI) para generar JSON final...")
    raw_txt_path = OUTPUT_DIR / f"newsletter_raw_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    generated_text = None
    try:
        generated_text = call_openai_chat(prompt, model="gpt-4o-mini", temperature=0.10, max_tokens=1800)
    except Exception as e:
        print("ERROR llamando al modelo:", e)
        generated_text = None

    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(generated_text or "")
    print("  -> Texto bruto guardado en:", raw_txt_path)

    # Intentar parsear / reparar -> parsed (dict)
    parsed = ensure_structured_json(generated_text or "", final_items)

    # ---------------------------------------------------
    # REGENERAR RESUMEN EJECUTIVO SI ES DÉBIL
    # ---------------------------------------------------
    try:
        current_summary = parsed.get("summary_exec", "")
        if is_summary_weak(current_summary):
            print("Resumen ejecutivo débil — generando versión mejorada...")
            # CORRECCIÓN: pasar final_items al regenerador
            new_summary = regenerate_summary_exec(parsed, final_items)
            if new_summary and not is_summary_weak(new_summary):
                parsed["summary_exec"] = new_summary
                print("✓ Nuevo resumen ejecutivo insertado.")
            else:
                print("El modelo no generó un mejor resumen, se conserva el original.")
    except Exception as e:
        print("Error en regeneración del resumen ejecutivo:", e)

    # Copiar logo
    copy_assets()

    # Renderizar HTML final
    html_path = OUTPUT_DIR / f"newsletter_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.html"
    try:
        out = render_html_from_struct(parsed, html_path)
        print("3) HTML generado en:", out)

    # Forzar index.html para GitHub Pages
        index_path = OUTPUT_DIR / "index.html"
        shutil.copyfile(html_path, index_path)
        print("✓ index.html generado para GitHub Pages")

    except Exception as e:
        print("Error al renderizar HTML:", e)
        print("Texto bruto guardado en:", raw_txt_path)

if __name__ == "__main__":
    main()