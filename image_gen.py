# image_gen.py
"""
Respaldo de imagen para las noticias cuyo artículo original no trae
og:image utilizable. Soporta tres proveedores, en este orden de
prioridad (el primero que tengas configurado gana):

1. Unsplash: busca una fotografía real de alta calidad por palabra clave
   de categoría. Gratis, sin tarjeta, sin riesgo de derechos de autor
   (fotos con licencia libre de Unsplash, con atribución). Es el
   recomendado por default.
2. Gemini (gemini-3.1-flash-image): genera una ilustración con IA.
3. OpenAI (gpt-image-1): genera una ilustración con IA.
   (2 y 3 requieren facturación activada en esas plataformas.)

Decisiones de diseño:
- Solo se usa cuando enrich.py no encontró imagen real en el artículo —
  nunca reemplaza la foto propia del artículo.
- Se guarda directamente en output/newsletters/editions/<fecha>/generated/
  y se referencia con la URL pública final de GitHub Pages, para que el
  mismo campo `image_url` funcione igual en el email y en la web (sin
  rutas relativas especiales).
"""
import base64
import hashlib

import requests

import config

# --------------------------------------------------------------------
# Unsplash (búsqueda de foto real por categoría)
# --------------------------------------------------------------------
_CATEGORY_SEARCH_QUERIES = {
    "competencia": "aerial drone mapping data",
    "equipos_geoespaciales": "lidar sensor drone survey",
    "procesamiento_ia": "point cloud 3d scanning technology",
    "construccion_innovacion": "construction site technology aerial",
    "tendencias_regulacion": "drone technology industry",
}
_DEFAULT_QUERY = "drone surveying technology"

# --------------------------------------------------------------------
# Generación con IA (respaldo de respaldo)
# --------------------------------------------------------------------
_CATEGORY_SCENE_HINTS = {
    "competencia": (
        "an abstract editorial illustration of drone mapping software: a glowing "
        "aerial data dashboard with orthomosaic tiles and flight paths over a "
        "construction site, viewed from above"
    ),
    "equipos_geoespaciales": (
        "a professional editorial illustration of a survey-grade drone equipped "
        "with a LiDAR sensor, scanning a landscape, with faint point-cloud scan "
        "lines visible in the air"
    ),
    "procesamiento_ia": (
        "an abstract visualization of a 3D point cloud being processed into a "
        "clean mesh by an AI pipeline, glowing data nodes and wireframe geometry "
        "in a dark studio background"
    ),
    "construccion_innovacion": (
        "an editorial illustration of a construction site with a translucent "
        "glowing digital twin wireframe overlay on a building under construction"
    ),
    "tendencias_regulacion": (
        "an abstract editorial illustration of a geospatial data grid overlaying "
        "a city skyline at dusk, conveying technology and regulation of aerial "
        "data"
    ),
}
_DEFAULT_SCENE = (
    "an abstract editorial illustration representing drone-based geospatial "
    "surveying technology"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 DAUCH-Newsletter-Bot"
    )
}


def _slug(item: dict) -> str:
    basis = item.get("link") or item.get("title") or ""
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _build_prompt(item: dict) -> str:
    scene = _CATEGORY_SCENE_HINTS.get(item.get("category"), _DEFAULT_SCENE)
    return (
        f"Professional editorial magazine illustration, {scene}. "
        "Clean minimalist tech style, dramatic lighting, color palette dominated "
        "by navy blue (#1A428A) and electric blue (#2E51EA) accents on a dark "
        "background. High detail, 16:9 composition. "
        "Absolutely no text, no logos, no brand names, no readable UI labels, "
        "no real company products, no human faces."
    )


def _search_unsplash(item: dict):
    """Devuelve (image_bytes, extension, credit_dict) o None."""
    query = _CATEGORY_SEARCH_QUERIES.get(item.get("category"), _DEFAULT_QUERY)
    auth = {"Authorization": f"Client-ID {config.UNSPLASH_ACCESS_KEY}"}

    resp = requests.get(
        "https://api.unsplash.com/search/photos",
        headers=auth,
        params={
            "query": query,
            "per_page": 10,
            "orientation": "landscape",
            "content_filter": "high",
        },
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        return None

    # Distribuye entre los resultados para no repetir la misma foto en
    # todas las noticias de una misma categoría dentro de una edición.
    photo = results[int(_slug(item), 16) % len(results)]

    img_resp = requests.get(photo["urls"]["regular"], headers=HEADERS, timeout=15)
    img_resp.raise_for_status()

    # Ping de tracking requerido por las guías de la API de Unsplash al
    # usar una foto (best-effort, no debe tumbar el flujo si falla).
    try:
        requests.get(photo["links"]["download_location"], headers=auth, timeout=8)
    except Exception:
        pass

    credit = {
        "name": photo.get("user", {}).get("name", "Unsplash"),
        "url": (
            photo.get("user", {}).get("links", {}).get("html", "https://unsplash.com")
            + "?utm_source=dauch_newsletter&utm_medium=referral"
        ),
    }
    return img_resp.content, "jpg", credit


def _generate_with_gemini(item: dict):
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_IMAGE_MODEL,
        contents=_build_prompt(item),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data, "png", None
    return None


def _generate_with_openai(item: dict):
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.images.generate(
        model=config.OPENAI_IMAGE_MODEL,
        prompt=_build_prompt(item),
        size="1536x1024",
        quality="medium",
        n=1,
    )
    b64 = response.data[0].b64_json
    if not b64:
        return None
    return base64.b64decode(b64), "png", None


def generate_fallback_image(item: dict, date_str: str) -> dict | None:
    """Busca/genera y guarda la imagen de respaldo. Devuelve
    {"image_url", "credit_name", "credit_url"} (los dos últimos None si no
    aplica) o None si no hay proveedor configurado o todos fallaron. Nunca
    lanza excepción — es un respaldo opcional."""
    providers = [
        ("Unsplash", config.UNSPLASH_ACCESS_KEY, _search_unsplash),
        ("Gemini", config.GEMINI_API_KEY, _generate_with_gemini),
        ("OpenAI", config.OPENAI_API_KEY, _generate_with_openai),
    ]

    for name, key, fn in providers:
        if not key:
            continue
        try:
            result = fn(item)
        except Exception as e:
            print(f"  ⚠️  {name} falló al buscar/generar imagen de respaldo: {e}")
            continue
        if not result:
            continue

        image_bytes, ext, credit = result
        out_dir = config.EDITIONS_DIR / date_str / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_slug(item)}.{ext}"
        (out_dir / filename).write_bytes(image_bytes)

        return {
            "image_url": f"{config.public_base_url()}/editions/{date_str}/generated/{filename}",
            "credit_name": credit["name"] if credit else None,
            "credit_url": credit["url"] if credit else None,
        }

    return None
