# run_newsletter.py
"""
Orquestador único del pipeline: fetch -> curate -> enrich -> editorial ->
render -> guardar -> enviar.

Uso:
    python run_newsletter.py --dry-run   # genera todo, no envía nada (default)
    python run_newsletter.py --test      # envía solo a config.TEST_RECIPIENT
    python run_newsletter.py --full      # envía a config.RECEIVERS (uso del cron)
"""
import argparse
from datetime import datetime, timezone

import config
import render
import send_email
from curate import curate
from editorial import build_newsletter
from enrich import enrich_items
from fetch_news import fetch_news


def next_edition_number() -> int:
    editions = render.list_past_editions()
    if not editions:
        return 1
    return max((e.get("edition_number") or 0) for e in editions) + 1


def build_plain_text(nl, web_url: str) -> str:
    lines = [
        "DAUCH — Novedades del sector geoespacial",
        f"Edición #{nl.edition_number} · {nl.date}",
        "",
        "RESUMEN EJECUTIVO",
        nl.summary_exec,
        "",
    ]
    for _key, label, items in render.sections(nl):
        lines.append(label.upper())
        for it in items:
            lines.append(f"- {it.title} ({it.source_name}): {it.source_link}")
        lines.append("")
    lines.append("RECOMENDACIÓN DAUCH")
    lines.append(nl.recommendation)
    lines.append("")
    lines.append(f"Ver edición interactiva (con imágenes y video): {web_url}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Genera y opcionalmente envía el newsletter DAUCH.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Genera pero no envía nada (default).")
    group.add_argument("--test", action="store_true", help=f"Envía solo a {config.TEST_RECIPIENT}.")
    group.add_argument("--full", action="store_true", help="Envía a la lista real de RECEIVERS.")
    args = parser.parse_args()
    mode = "test" if args.test else ("full" if args.full else "dry-run")

    config.require_anthropic_key()

    print("1) Descargando noticias de todas las fuentes...")
    raw_items = fetch_news()
    print(f"   -> {len(raw_items)} noticias descargadas en bruto.")

    print("2) Curando (dedupe + filtro duro + clasificación con Claude Haiku)...")
    final_items = curate(raw_items)
    if not final_items:
        print("❌ No quedaron noticias tras la curación. Abortando sin generar edición.")
        return

    print("3) Enriqueciendo medios (imagen/video del artículo original)...")
    final_items = enrich_items(final_items)

    print("4) Redactando la edición con Claude Opus 5...")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    edition_number = next_edition_number()
    nl = build_newsletter(final_items, edition_number, date_str)

    print("5) Renderizando y guardando la edición...")
    render.copy_assets()
    web_url = f"{config.public_base_url()}/editions/{date_str}/index.html"
    email_html = render.render_email(nl, web_url)
    meta = render.save_edition(nl, email_html)
    print(f"   -> Edición #{edition_number} guardada en output/newsletters/{meta['url']}")
    print(f"   -> Vista previa del email: output/newsletters/email_preview_{date_str}.html")

    print(f"6) Envío de correo (modo: {mode})...")
    plain_text = build_plain_text(nl, web_url)
    subject = f"{config.SUBJECT_PREFIX} · {date_str}"
    sent = send_email.send_newsletter_email(email_html, plain_text, subject, mode)
    if sent:
        print("✅ Newsletter enviado.")
    else:
        print("ℹ️  Newsletter generado, no se envió correo (revisa el modo usado).")


if __name__ == "__main__":
    main()
