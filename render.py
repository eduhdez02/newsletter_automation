# render.py
"""
Renderiza el mismo objeto Newsletter (editorial.Newsletter) en varias
salidas, todas a partir de los mismos datos:

- email_newsletter.html: compatible con clientes de correo (tablas, CSS
  inline, sin animaciones ni video). send_email.py convierte sus
  "assets/logo-*.png" a "cid:logo-*" al armar el MIME real.
- web_newsletter.html: versión rica para GitHub Pages (tema oscuro,
  animaciones, video embebido) — guardada de forma permanente en
  editions/<fecha>/index.html.
- redirect.html: index.html raíz, redirige a la última edición.
- archive_index.html: archive.html raíz, lista todas las ediciones.

Las plantillas web usan un asset path relativo (`assets`) que cambia según
la profundidad del archivo final (raíz vs. editions/<fecha>/), por eso cada
`render_*` recibe explícitamente su prefijo de assets.
"""
import json
import re
import shutil

from jinja2 import Environment, FileSystemLoader

import config
from editorial import Newsletter

env = Environment(loader=FileSystemLoader(str(config.TEMPLATE_DIR)))

CATEGORY_LABELS = {
    "competencia": "Competencia",
    "equipos_geoespaciales": "Equipos geoespaciales",
    "procesamiento_ia": "Procesamiento & IA",
    "construccion_innovacion": "Construcción e innovación",
    "tendencias_regulacion": "Tendencias & regulación",
}

CATEGORY_ORDER = list(CATEGORY_LABELS.keys())

_YOUTUBE_ID_RE = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/))([\w-]{6,15})")


def youtube_embed_url(video_url: str | None) -> str | None:
    if not video_url:
        return None
    m = _YOUTUBE_ID_RE.search(video_url)
    return f"https://www.youtube.com/embed/{m.group(1)}" if m else None


env.globals["youtube_embed_url"] = youtube_embed_url


def sections(nl: Newsletter):
    """Lista pública de (category_key, label, items) en el orden fijo del
    newsletter, saltando categorías vacías. La usan las plantillas y
    run_newsletter.py (para el fallback de texto plano del email)."""
    out = []
    for key in CATEGORY_ORDER:
        items = getattr(nl, key)
        if items:
            out.append((key, CATEGORY_LABELS[key], items))
    return out


# Alias retrocompatible por si algo interno todavía llama a la versión
# "privada".
_sections = sections


def render_email(nl: Newsletter, web_url: str) -> str:
    tpl = env.get_template("email_newsletter.html")
    return tpl.render(
        nl=nl, sections=_sections(nl), web_url=web_url,
        from_email=config.FROM_EMAIL, assets="assets",
    )


def render_web(nl: Newsletter, archive_url: str, assets: str = "../../assets") -> str:
    tpl = env.get_template("web_newsletter.html")
    return tpl.render(nl=nl, sections=_sections(nl), archive_url=archive_url, assets=assets)


def render_redirect(latest_url: str, assets: str = "assets") -> str:
    tpl = env.get_template("redirect.html")
    return tpl.render(latest_url=latest_url, assets=assets)


def render_archive_index(editions: list[dict], assets: str = "assets") -> str:
    tpl = env.get_template("archive_index.html")
    return tpl.render(editions=editions, assets=assets)


def copy_assets():
    """Copia logos y fuentes a output/newsletters/assets para que tanto la
    edición actual como el archivo histórico los puedan servir."""
    src = config.TEMPLATE_DIR / "assets"
    dst = config.OUTPUT_DIR / "assets"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def list_past_editions() -> list[dict]:
    """Lee output/newsletters/editions/*/meta.json para construir el
    índice del archivo histórico."""
    editions = []
    if not config.EDITIONS_DIR.exists():
        return editions
    for d in sorted(config.EDITIONS_DIR.iterdir(), reverse=True):
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                editions.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                continue
    return editions


def save_edition(nl: Newsletter, email_html: str) -> dict:
    """Guarda la edición actual bajo editions/<fecha>/ (index.html +
    meta.json) SIN borrar ediciones anteriores, regenera el index.html raíz
    (redirect a la última) y archive.html (listado completo). Devuelve el
    meta dict de la edición guardada."""
    edition_dir = config.EDITIONS_DIR / nl.date
    edition_dir.mkdir(parents=True, exist_ok=True)

    web_html = render_web(nl, archive_url="../../archive.html", assets="../../assets")
    (edition_dir / "index.html").write_text(web_html, encoding="utf-8")

    meta = {
        "date": nl.date,
        "edition_number": nl.edition_number,
        "summary_exec": nl.summary_exec,
        "url": f"editions/{nl.date}/index.html",
    }
    (edition_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # index.html raíz = redirección a la última edición (no un espejo
    # pesado con rutas de assets ambiguas).
    redirect_html = render_redirect(latest_url=meta["url"], assets="assets")
    (config.OUTPUT_DIR / "index.html").write_text(redirect_html, encoding="utf-8")

    # archive.html con el listado completo de ediciones pasadas.
    archive_html = render_archive_index(list_past_editions(), assets="assets")
    (config.OUTPUT_DIR / "archive.html").write_text(archive_html, encoding="utf-8")

    # Vista previa del email (para --dry-run): mismo nivel que /assets para
    # que las rutas relativas de los logos resuelvan bien en el navegador.
    (config.OUTPUT_DIR / f"email_preview_{nl.date}.html").write_text(
        email_html, encoding="utf-8"
    )

    return meta
