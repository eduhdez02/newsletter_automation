# send_email.py
"""
Envío del newsletter como email HTML real (multipart/related: alternative
text/plain + text/html, con los logos DAUCH/DAURIX embebidos vía Content-ID
para que carguen siempre, sin depender de que el cliente de correo permita
imágenes remotas).

Reemplaza el envío anterior (texto plano con un link a GitHub Pages) que se
veía poco profesional.

Modos de uso (ver run_newsletter.py para el flujo completo):
- --dry-run: no envía nada.
- --test: envía SOLO a config.TEST_RECIPIENT (por ahora, eduardohernandez@dauch.com.mx).
- --full: envía a config.RECEIVERS (lista real) — uso exclusivo del cron programado.
"""
import argparse
import smtplib
import ssl
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import config

LOGO_CIDS = {
    "logo-dauch": config.TEMPLATE_DIR / "assets" / "logo-dauch-navy.png",
    "logo-daurix": config.TEMPLATE_DIR / "assets" / "logo-daurix-navy.png",
}


def _inline_logos(html: str) -> str:
    for cid in LOGO_CIDS:
        html = html.replace(f'src="assets/{cid}-navy.png"', f'src="cid:{cid}"')
    return html


def build_message(html: str, plain_text: str, subject: str, recipients: list[str]) -> MIMEMultipart:
    msg = MIMEMultipart("related")
    msg["From"] = f"{config.FROM_NAME} <{config.FROM_EMAIL}>"
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plain_text, "plain", "utf-8"))
    alt.attach(MIMEText(_inline_logos(html), "html", "utf-8"))
    msg.attach(alt)

    for cid, path in LOGO_CIDS.items():
        if not path.exists():
            continue
        with open(path, "rb") as f:
            img = MIMEImage(f.read(), _subtype="png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=path.name)
        msg.attach(img)

    return msg


def send_message(msg: MIMEMultipart, recipients: list[str]):
    config.require_smtp_credentials()
    context = ssl.create_default_context()
    print(f"📨 Conectando a SMTP {config.SMTP_HOST}:{config.SMTP_PORT} ...")
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.ehlo()
        if config.SMTP_PORT == 587:
            server.starttls(context=context)
            server.ehlo()
        server.login(config.SMTP_USER, config.SMTP_PASS)
        server.sendmail(config.FROM_EMAIL, recipients, msg.as_string())
    print("✅ Correo enviado a:", recipients)


def send_newsletter_email(html: str, plain_text: str, subject: str, mode: str) -> bool:
    """mode: 'dry-run' | 'test' | 'full'. Devuelve True si se envió algo."""
    if mode == "dry-run":
        print("🧪 --dry-run: no se envía ningún correo.")
        return False

    if mode == "test":
        recipients = [config.TEST_RECIPIENT]
    elif mode == "full":
        recipients = config.RECEIVERS
        if not recipients:
            print("❌ RECEIVERS está vacío. No se puede enviar en modo --full.")
            return False
    else:
        raise ValueError(f"Modo desconocido: {mode}")

    msg = build_message(html, plain_text, subject, recipients)
    send_message(msg, recipients)
    return True


# --------------------------------------------------------------------
# CLI standalone: reenvía la última vista previa generada, sin recalcular
# el pipeline completo (útil para reenviar manualmente).
# --------------------------------------------------------------------
def _find_latest_preview() -> Path | None:
    previews = sorted(config.OUTPUT_DIR.glob("email_preview_*.html"), reverse=True)
    return previews[0] if previews else None


def main():
    parser = argparse.ArgumentParser(description="Reenvía la última vista previa de email generada.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()

    mode = "dry-run" if args.dry_run else ("test" if args.test else "full")

    latest = _find_latest_preview()
    if not latest:
        print("❌ No hay ninguna vista previa de email en output/newsletters/. Corre run_newsletter.py primero.")
        return

    html = latest.read_text(encoding="utf-8")
    date_str = latest.stem.replace("email_preview_", "")
    web_url = f"{config.public_base_url()}/editions/{date_str}/index.html"
    plain_text = (
        "Nueva edición del newsletter DAUCH disponible.\n\n"
        f"Ver edición completa: {web_url}\n"
    )
    subject = f"{config.SUBJECT_PREFIX} · {date_str}"

    send_newsletter_email(html, plain_text, subject, mode)


if __name__ == "__main__":
    main()
