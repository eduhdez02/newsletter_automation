# config.py
"""
Punto único de configuración del proyecto.

Carga variables de entorno (.env en local, GitHub Secrets en Actions),
valida las obligatorias y expone nombres consistentes en todo el pipeline
(antes había una mezcla de EMAIL_PASS / SMTP_PASS / SMTP_PASSWORD que no
coincidían entre .env y send_email.py).
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# En Windows la consola por default usa cp1252, que no puede imprimir los
# emojis usados en los mensajes de progreso (❌ ✅ ⚠️). Forzamos UTF-8 en
# stdout/stderr para que el pipeline no truene al hacer print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()


def _env(name: str, default: str = "") -> str:
    """os.getenv, pero trata '' como 'no definida'.

    En GitHub Actions, un `env: { X: ${{ secrets.X }} }` con el secret sin
    configurar define la variable igual, con valor "" — no la deja ausente.
    os.getenv(name, default) solo aplica el default cuando la variable no
    existe en absoluto, así que un secret opcional sin configurar pisaría
    silenciosamente el default (o, en el caso de SMTP_PORT, tronaría el
    int() de una cadena vacía). Esta función hace que "" cuente como "usa
    el default" en todo el proyecto.
    """
    val = os.getenv(name)
    return val if val else default


ROOT = Path(__file__).parent.resolve()
TEMPLATE_DIR = ROOT / "templates"
OUTPUT_DIR = ROOT / "output" / "newsletters"
EDITIONS_DIR = OUTPUT_DIR / "editions"

# --------------------------------------------------------------------
# IA (Anthropic / Claude)
# --------------------------------------------------------------------
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
CLASSIFIER_MODEL = _env("CLASSIFIER_MODEL", "claude-haiku-4-5")
EDITOR_MODEL = _env("EDITOR_MODEL", "claude-opus-5")

# Solo necesaria si tu API key NO está ligada a un workspace específico
# (personal/org key "multi-workspace"). En ese caso la API exige mandar el
# ID del workspace en cada request. Se obtiene en Console -> Settings ->
# Workspaces (columna ID). Si tu key ya está ligada a un workspace, deja
# esto vacío.
ANTHROPIC_WORKSPACE_ID = _env("ANTHROPIC_WORKSPACE_ID")


def anthropic_client_kwargs() -> dict:
    """kwargs comunes para construir el cliente Anthropic en curate.py y
    editorial.py, agregando el header de workspace solo si hace falta."""
    kwargs = {"api_key": ANTHROPIC_API_KEY}
    if ANTHROPIC_WORKSPACE_ID:
        kwargs["default_headers"] = {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID}
    return kwargs

# --------------------------------------------------------------------
# SMTP / correo
# --------------------------------------------------------------------
SMTP_HOST = _env("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_env("SMTP_PORT", "587"))
SMTP_USER = _env("SMTP_USER")
SMTP_PASS = _env("SMTP_PASS")

FROM_EMAIL = _env("FROM_EMAIL") or SMTP_USER
FROM_NAME = _env("FROM_NAME", "DAUCH")

SUBJECT_PREFIX = _env("SUBJECT_PREFIX", "DAUCH — Novedades del sector geoespacial")

# Lista real de destinatarios (se usa solo en modo --full)
RECEIVERS = [r.strip() for r in _env("RECEIVERS").split(",") if r.strip()]

# Destinatario único de pruebas (modo --test). Por ahora, solo Eduardo.
TEST_RECIPIENT = _env("TEST_RECIPIENT", "eduardohernandez@dauch.com.mx")

# --------------------------------------------------------------------
# GitHub Pages
# --------------------------------------------------------------------
GITHUB_REPOSITORY = _env("GITHUB_REPOSITORY")


def public_base_url() -> str:
    """URL pública de la versión web, agnóstica del owner (usuario u org)."""
    if GITHUB_REPOSITORY:
        owner, repo = GITHUB_REPOSITORY.split("/", 1)
        return f"https://{owner}.github.io/{repo}"
    return f"file://{OUTPUT_DIR.resolve()}"


def require_anthropic_key():
    if not ANTHROPIC_API_KEY:
        print(
            "❌ ANTHROPIC_API_KEY no está definida. "
            "Configúrala en .env (local) o en GitHub Secrets (Actions).",
            file=sys.stderr,
        )
        sys.exit(1)


def require_smtp_credentials():
    if not SMTP_USER or not SMTP_PASS:
        print(
            "❌ Faltan credenciales SMTP (SMTP_USER / SMTP_PASS). Abortando.",
            file=sys.stderr,
        )
        sys.exit(1)
