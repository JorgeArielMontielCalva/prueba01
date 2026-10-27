# prueba01
## Sitio ANTEVRIC

- `site/` — sitio publicado (HTML, CSS y JS puros). Netlify lo sirve tal cual (`netlify.toml`).
- `src/pages/` — contenido de cada página (solo lo que va dentro de `<main>`).
- `src/build.py` — agrega cabecera, menú, pie y `<head>` comunes y escribe en `site/`.

Para cambiar el menú, el pie o el `<head>`, edita `src/build.py`. Para cambiar una página, edita su archivo en `src/pages/`. Después ejecuta:

```
python3 src/build.py
```

y sube también los archivos generados en `site/`. No edites el HTML de `site/` directamente: se sobrescribe en la siguiente generación. CSS y JS sí se editan directo en `site/assets/`.
