# send_email.py
# Envío simple: solo link al newsletter (GitHub Pages)

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime
import sys

# =========================
# RUTAS
# =========================

ROOT = Path(__file__).parent.resolve()
NEWS_DIR = ROOT / "output" / "newsletters"

# =========================
# CONFIG SMTP / MAIL
# =========================

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS") or os.getenv("SMTP_PASSWORD")

FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER)
FROM_NAME = os.getenv("FROM_NAME", "DAUCH")

RECEIVERS = os.getenv("RECEIVERS", "")
SUBJECT_PREFIX = os.getenv("SUBJECT_PREFIX", "DAUCH — Newsletter")

if not SMTP_USER or not SMTP_PASS:
    print("❌ Faltan credenciales SMTP (SMTP_USER / SMTP_PASS). Abortando.")
    sys.exit(1)

# =========================
# UTILIDADES
# =========================

def find_latest_html():
    files = sorted(
        NEWS_DIR.glob("newsletter_*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return files[0] if files else None


def build_public_url(latest_html_path: Path):
    """
    Prioridad:
    1) GitHub Pages (en Actions)
    2) file:// en local
    """
    g_repo = os.getenv("GITHUB_REPOSITORY")
    if g_repo:
        owner, repo = g_repo.split("/", 1)
        return f"https://{owner}.github.io/{repo}/"

    return f"file://{latest_html_path.resolve()}"


def send_link_email(public_url: str, recipients: list[str]):
    date_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"{SUBJECT_PREFIX} · {date_str}"

    msg = EmailMessage()
    msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    body = (
        f"Hola,\n\n"
        f"Ya está disponible la edición quincenal del newsletter de DAUCH.\n\n"
        f"Puedes consultarlo aquí:\n"
        f"{public_url}\n\n"
        f"Saludos,\n"
        f"DAUCH"
    )
    msg.set_content(body)

    context = ssl.create_default_context()
    print(f"📨 Conectando a SMTP {SMTP_HOST}:{SMTP_PORT} ...")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        if SMTP_PORT == 587:
            server.starttls(context=context)
            server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

    print("✅ Correo enviado a:", recipients)


# =========================
# MAIN
# =========================

def main():
    latest = find_latest_html()
    if not latest:
        print("❌ No se encontró ningún newsletter HTML en:", NEWS_DIR)
        return

    public_url = build_public_url(latest)

    recipients = [r.strip() for r in RECEIVERS.split(",") if r.strip()]
    if not recipients:
        print("❌ No hay destinatarios. Define RECEIVERS en el entorno.")
        return

    print("🔗 URL pública del newsletter:", public_url)
    send_link_email(public_url, recipients)


if __name__ == "__main__":
    main()