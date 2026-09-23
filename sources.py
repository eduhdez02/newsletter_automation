# sources.py
"""
Taxonomía de contenido y fuentes del newsletter DAUCH.

Reemplaza la lista plana de 28 feeds genéricos de "drones" (RSS_FEEDS en el
.env original) por dos mecanismos:

1. Feeds directos curados: medios/blogs realmente enfocados en topografía,
   geoespacial, LiDAR, fotogrametría y construcción con tecnología de punta.
   Se eliminaron los feeds genéricos de drones (militar, consumo, racing,
   podcasts) que antes contaminaban la selección.

2. Búsquedas de Google News RSS por tema: no requieren API key y permiten
   cubrir temas específicos (competencia directa, software de procesamiento,
   equipos) sin depender de que esas empresas tengan RSS público. Google
   News ya filtra por relevancia de la búsqueda, lo que reduce muchísimo el
   ruido frente a un feed genérico.
"""

# Categorías finales del newsletter (usadas por curate.py y editorial.py)
CATEGORIES = {
    "competencia": (
        "Movimientos de competencia directa o cercana a DAUCH: DroneDeploy, "
        "Propeller Aero, Skycatch, Kespry, Reconstruct, OpenSpace, y "
        "plataformas de captura/gestión de datos geoespaciales para "
        "construcción, minería o infraestructura."
    ),
    "equipos_geoespaciales": (
        "Hardware nuevo o mejorado: antenas/receptores GNSS-RTK, sensores "
        "LiDAR, cámaras multiespectrales/térmicas, drones de levantamiento "
        "topográfico, estaciones base, escáneres 3D."
    ),
    "procesamiento_ia": (
        "Software de procesamiento de datos geoespaciales con nuevas "
        "capacidades de IA o automatización: DJI Terra, Pix4D, Agisoft "
        "Metashape, procesamiento de nubes de puntos, fotogrametría "
        "automatizada, clasificación automática de datos."
    ),
    "construccion_innovacion": (
        "Innovación en el sector construcción con tecnología de punta: "
        "reality capture, gemelos digitales (digital twins), BIM integrado "
        "con drones, monitoreo de avance de obra, automatización de "
        "procesos constructivos."
    ),
    "tendencias_regulacion": (
        "Tendencias de mercado, adopción de IA en el sector geoespacial/"
        "construcción, regulación de drones relevante para operaciones "
        "comerciales, inversión y consolidación de empresas del sector."
    ),
}

# --------------------------------------------------------------------
# Feeds directos curados (medios/blogs on-topic)
#
# Nota: varios blogs de la lista original (commercialuavnews.com,
# gim-international.com, sphengineering.com, datumate.com, aerotas.com,
# rockrobotic.com) cambiaron de plataforma y su RSS ya no responde (404).
# En vez de adivinar la nueva ruta, se cubren esos mismos dominios vía
# Google News con `site:` (ver _SITE_RESTRICTED_DOMAINS más abajo) — más
# robusto a futuros cambios de CMS que hardcodear una URL de feed.
# --------------------------------------------------------------------
DIRECT_FEEDS = [
    "https://lidarnews.com/feed/",
    "https://geospatialworld.net/feed",
    "https://digital-geography.com/feed",
    "https://dronelife.com/feed/",
    "https://geoconnexion.com/news/feed.rss",
]

_SITE_RESTRICTED_DOMAINS = [
    "commercialuavnews.com",
    "gim-international.com",
    "sphengineering.com",
    "datumate.com",
    "aerotas.com",
    "rockrobotic.com",
    "blickfeld.com",
]

# --------------------------------------------------------------------
# Búsquedas de Google News RSS (sin API key)
# --------------------------------------------------------------------
_GOOGLE_NEWS_QUERIES = [
    "DroneDeploy",
    "Propeller Aero",
    "Skycatch",
    "Kespry drone mapping",
    "OpenSpace reality capture",
    "DJI Terra IA OR AI",
    "Pix4D update",
    "Agisoft Metashape",
    "reality capture construction",
    "digital twin construction drone",
    "LiDAR survey drone",
    "GNSS RTK receiver survey",
    "point cloud processing AI",
    "geospatial AI construction",
    "drone topografía levantamiento",
]


def google_news_rss(query: str, lang: str = "en-US", country: str = "US") -> str:
    """Construye una URL de Google News RSS para una búsqueda puntual."""
    from urllib.parse import quote

    ceid = f"{country}:{lang.split('-')[0]}"
    return (
        "https://news.google.com/rss/search?q="
        f"{quote(query)}&hl={lang}&gl={country}&ceid={ceid}"
    )


def google_news_feeds() -> list[str]:
    queries = list(_GOOGLE_NEWS_QUERIES)
    queries += [f"site:{domain}" for domain in _SITE_RESTRICTED_DOMAINS]
    return [google_news_rss(q) for q in queries]


def all_feeds() -> list[str]:
    return DIRECT_FEEDS + google_news_feeds()


# --------------------------------------------------------------------
# Filtro duro de exclusión (antes del LLM, sin costo)
# --------------------------------------------------------------------
NEGATIVE_KEYWORDS = [
    "missile", "airstrike", "air strike", "shahed", "combat drone",
    "warfare", "military strike", "hostage", "war zone",
    "racing drone", "fpv freestyle", "drone racing", "wedding photography",
    "drone light show", "toy drone", "drone review 2025 gift",
]

# Si el título/resumen matchea alguna de estas, NO se descarta aunque
# también matchee una negative keyword (ej. "regulación de drones militares
# afecta certificación comercial" sí puede interesar).
POSITIVE_OVERRIDE_KEYWORDS = [
    "survey", "topograf", "geoespacial", "geospatial", "lidar", "gnss",
    "rtk", "fotogrametr", "photogrammetry", "point cloud", "nube de puntos",
    "construction", "construcción", "reality capture", "digital twin",
    "bim ",
]
