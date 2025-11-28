# send_email.py
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).parent.resolve()
NEWS_DIR = ROOT / "output" / "newsletters"

# Leer variables de entorno (en GitHub Actions las definirás como secrets/env)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")  # app password o API key
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER)
RECEIVERS = os.getenv("RECEIVERS", "")  # coma-separados
SUBJECT_PREFIX = os.getenv("SUBJECT_PREFIX", "DAUCH — Newsletter")

if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
    print("Faltan variables SMTP en el entorno (SMTP_HOST/SMTP_USER/SMTP_PASS). Abortando.")
    sys.exit(1)

def find_latest_html():
    files = sorted(NEWS_DIR.glob("newsletter_*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def html_to_text(html_path):
    # fallback simple: extraer texto plano mínimo si el cliente no muestra HTML
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return ""
    with open(html_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    soup = BeautifulSoup(html, "html.parser")
    # extraer los primeros 6 párrafos como texto plano
    paragraphs = soup.find_all("p")
    txt = "\n\n".join([p.get_text().strip() for p in paragraphs[:6]])
    return txt[:4000]

def send_html_email(html_path, recipients):
    date_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"{SUBJECT_PREFIX} · {date_str}"

    msg = EmailMessage()
    msg["From"] = FROM_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(html_to_text(html_path) or "Adjunto: newsletter HTML")
    # Leer HTML como cuerpo alternativo
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    msg.add_alternative(html_content, subtype="html")

    context = ssl.create_default_context()
    print(f"Conectando a SMTP {SMTP_HOST}:{SMTP_PORT} ...")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        if SMTP_PORT == 587:
            server.starttls(context=context)
            server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    print("Correo enviado a:", recipients)

def main():
    latest = find_latest_html()
    if not latest:
        print("No se encontró HTML en", NEWS_DIR)
        return
    print("Enviando archivo:", latest)
    recs = [r.strip() for r in RECEIVERS.split(",") if r.strip()]
    if not recs:
        print("No hay destinatarios. Define RECEIVERS en el entorno (coma-separated).")
        return
    send_html_email(latest, recs)

if __name__ == "__main__":
    main()