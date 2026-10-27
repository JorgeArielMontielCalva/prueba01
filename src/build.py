"""Ensambla las páginas de site/ a partir de src/pages/ más la cabecera y el pie comunes.

Uso (desde la raíz del repo):  python3 src/build.py
El HTML generado se versiona; Netlify publica site/ tal cual, sin paso de build.
"""
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "site"

NAV = [
    ("index.html", "00", "Inicio"),
    ("servicios.html", "01", "Servicios"),
    ("metodo.html", "02", "Método"),
    ("trabajo.html", "03", "Trabajo"),
    ("estudio.html", "04", "Estudio"),
]

PAGES = {
    "index.html": ("ANTEVRIC — Desarrollo · Diseño · Infraestructura",
                   "Construyo la infraestructura digital de tu negocio: sitio a medida, correo con dominio propio, identidad, campañas y soporte. Estudio técnico en México."),
    "servicios.html": ("Servicios — ANTEVRIC",
                       "Desarrollo e infraestructura, diseño e identidad, presencia y campañas, soporte, investigación y equipo. Lo que hago y lo que no."),
    "metodo.html": ("Método e inversión — ANTEVRIC",
                    "Valoración gratuita de 30 minutos, cotización por escrito, anticipo del 50 % y un proyecto a la vez. Rangos de inversión y preguntas frecuentes."),
    "trabajo.html": ("Trabajo — ANTEVRIC",
                     "Caso: campañas en Meta Ads para tres ramas de una empresa de mantenimiento industrial."),
    "estudio.html": ("Estudio — ANTEVRIC",
                     "ANTEVRIC es un estudio técnico de una sola persona: Jorge Montiel, desarrollador."),
    "contacto.html": ("Agenda tu valoración — ANTEVRIC",
                      "Agenda una valoración gratuita de 30 minutos. Termina en una cotización escrita."),
    "gracias.html": ("Solicitud recibida — ANTEVRIC", "Tu solicitud llegó. Te contacto para agendar la valoración."),
    "404.html": ("Ruta no encontrada — ANTEVRIC", "Esta página no existe."),
}

LOGO = """<a class="logo" href="/" aria-label="ANTEVRIC, inicio">
        <!-- [PENDIENTE: reemplaza .logo-slot por el SVG del logotipo] -->
        <span class="logo-slot" aria-hidden="true">SVG</span>
        <span class="logo-word">ANTEVRIC</span>
      </a>"""


def header(cur):
    items = []
    for href, n, name in NAV:
        ac = ' aria-current="page"' if href == cur else ""
        link = "/" if href == "index.html" else "/" + href
        items.append(f'<li><a href="{link}"{ac}><span class="mono">{n}</span>{name}</a></li>')
    items = "\n          ".join(items)
    return f"""<a class="skip" href="#main">Saltar al contenido</a>
  <div class="topbar mono">
    <div class="wrap">
      <span><span class="status-dot" aria-hidden="true"></span>Valoración gratuita · 30 min</span>
      <span class="hide-sm">Desarrollo · Diseño · Infraestructura</span>
      <span>MX <span data-clock>--:--:--</span></span>
    </div>
  </div>
  <header class="site-header">
    <div class="wrap">
      {LOGO}
      <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="nav"><span class="t">Menú</span><span class="bars" aria-hidden="true"></span></button>
      <nav class="nav" id="nav" aria-label="Principal">
        <ul>
          {items}
        </ul>
        <div class="nav-cta"><a class="btn" href="/contacto.html">Agenda tu valoración <span class="arr" aria-hidden="true">→</span></a></div>
      </nav>
    </div>
  </header>"""


FOOTER = """<footer class="site-footer">
    <div class="wrap">
      <div class="footer-grid">
        <div class="footer-brand">
          <span class="label">Estudio técnico · México</span>
          <span class="logo-word">ANTEVRIC</span>
          <p>Desarrollo · Diseño · Infraestructura</p>
        </div>
        <div>
          <h4>Mapa</h4>
          <ul>
            <li><a href="/servicios.html">Servicios</a></li>
            <li><a href="/metodo.html">Método</a></li>
            <li><a href="/trabajo.html">Trabajo</a></li>
            <li><a href="/estudio.html">Estudio</a></li>
          </ul>
        </div>
        <div>
          <h4>Contacto</h4>
          <ul>
            <li><a href="/contacto.html">Valoración</a></li>
            <li><span class="pending">[PENDIENTE: correo]</span></li>
            <li><span class="pending">[PENDIENTE: WhatsApp]</span></li>
          </ul>
        </div>
        <div>
          <h4>Redes</h4>
          <ul>
            <li><span class="pending">[PENDIENTE: enlaces a redes]</span></li>
          </ul>
        </div>
      </div>
      <div class="footer-giant" aria-hidden="true">ANTEVRIC</div>
      <div class="footer-base mono">
        <span>© <span data-year>2026</span> ANTEVRIC</span>
        <span>Un proyecto a la vez</span>
      </div>
    </div>
  </footer>"""


def page(fname, body):
    title, desc = PAGES[fname]
    extra = '\n  <meta name="robots" content="noindex">' if fname in ("gracias.html", "404.html") else ""
    return f"""<!doctype html>
<html lang="es-MX">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <meta name="description" content="{desc}">
  <meta name="theme-color" content="#141719">{extra}
  <meta property="og:title" content="{title}">
  <meta property="og:description" content="{desc}">
  <meta property="og:type" content="website">
  <!-- [PENDIENTE: og:image y favicon con el logotipo] -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap">
  <link rel="stylesheet" href="/assets/styles.css">
  <script>document.documentElement.classList.add("js");</script>
</head>
<body>
  {header(fname)}

  <main id="main">
{body}
  </main>

  {FOOTER}
  <script src="/assets/main.js" defer></script>
</body>
</html>
"""


for f in PAGES:
    body = (HERE / "pages" / f).read_text()
    (OUT / f).write_text(page(f, body))
    print("ok", f)
