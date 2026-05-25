import httpx
from langchain.messages import ToolMessage
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime

MONUMENTS_URL = "https://fly.io/apps/backend-tfg/api/v1/public/monuments"
ROUTES_URL = "https://fly.io/apps/backend-tfg/api/v1/public/route"


@tool
def get_monuments(runtime: ToolRuntime) -> str:
    """Obtiene la lista completa de monumentos disponibles desde el backend."""
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(MONUMENTS_URL)
            response.raise_for_status()
            data = response.json()

        resumen_lineas = []
        for m in data:
            sinopsis = next(
                (d["contenido"] for d in m.get("description", [])
                 if d.get("language") == "es" and not d.get("kids") and not d.get("complete")),
                "Sin sinopsis."
            )
            resumen_lineas.append(
                f"- ID: {m['id']} | Nombre: {m['name']} | Categoría: {m['tag']['name']} | "
                f"Likes: {m['NLikes']} | Accesible: {m['accessibility']}\n"
                f"  Sinopsis: {sinopsis}"
            )

        return "MONUMENTOS DISPONIBLES:\n" + "\n".join(resumen_lineas)

    except httpx.HTTPStatusError as e:
        return f"Error HTTP al obtener monumentos: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        return f"Error de conexión con el endpoint de monumentos: {str(e)}"
    except Exception as e:
        return f"Error inesperado en get_monuments: {str(e)}"


import json # Añade esto arriba si no lo tienes

@tool
def get_monument_detail(runtime: ToolRuntime) -> str:
    """Obtiene el detalle completo del monumento registrado en el estado actual."""
    monument_id   = runtime.state.get("monument_id")

    if not monument_id:
        return (
            "ERROR: No se ha registrado ningún monumento en el estado. "
            "Usa primero 'update_monument_state' para indicar el monumento."
        )

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(MONUMENTS_URL)
            response.raise_for_status()
            data = response.json()

        monumento = next((m for m in data if m["id"] == monument_id), None)

        if not monumento:
            return f"No se encontró ningún monumento con ID '{monument_id}' en el backend."

        # Devolvemos el JSON íntegro. La IA es perfectamente capaz de leerlo y extraer
        # la información para su análisis, y además tendrá las URLs intactas.
        return json.dumps(monumento, ensure_ascii=False)

    except httpx.HTTPStatusError as e:
        return f"Error HTTP al obtener detalle del monumento: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        return f"Error de conexión: {str(e)}"
    except Exception as e:
        return f"Error inesperado en get_monument_detail: {str(e)}"

@tool
def get_routes(runtime: ToolRuntime) -> str:
    """Obtiene la lista completa de rutas turísticas disponibles desde el backend."""
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(ROUTES_URL)
            response.raise_for_status()
            data = response.json()

        dificultad_label = {0: "Fácil", 1: "Media", 2: "Difícil"}

        resumen_lineas = []
        for r in data:
            tiempo_min = round(r.get("estimated_time_seconds", 0) / 60, 1)
            distancia_km = round(r.get("total_distance_meters", 0) / 1000, 2)

            # Mapeo de campos modificados
            diff_val = r.get("difficult", r.get("difficulty", 0))
            dif = dificultad_label.get(diff_val, str(diff_val))
            activa = "Sí" if r.get("isActive", r.get("activate", False)) else "No"
            valoracion = r.get("average_score", r.get("rating", 0.0))

            # Procesar el nuevo array de objetos monumento
            monumentos_lista = r.get("monuments", [])
            monumentos_nombres = []
            for m in monumentos_lista:
                if isinstance(m, dict):
                    monumentos_nombres.append(f"{m.get('name', 'Desconocido')} (ID: {m.get('id', '')})")
                else:
                    monumentos_nombres.append(str(m))

            monumentos_str = ", ".join(monumentos_nombres) if monumentos_nombres else "Ninguno"

            resumen_lineas.append(
                f"- ID: {r['id']} | Nombre: {r['name']} | Dificultad: {dif} | "
                f"Valoración: {valoracion}/5 | Activa: {activa}\n"
                f"  Distancia: {distancia_km} km | Tiempo estimado: {tiempo_min} min\n"
                f"  Descripción: {r['description']}\n"
                f"  Monumentos incluidos: {monumentos_str}"
            )

        return "RUTAS DISPONIBLES:\n" + "\n".join(resumen_lineas)

    except httpx.HTTPStatusError as e:
        return f"Error HTTP al obtener rutas: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        return f"Error de conexión con el endpoint de rutas: {str(e)}"
    except Exception as e:
        return f"Error inesperado en get_routes: {str(e)}"


@tool
def update_monument_state(
        monument_id: str,
        full_monument_name: str,
        tag: str,
        runtime: ToolRuntime,
) -> Command:
    """
    Registra en el estado el monumento sobre el que va a trabajar el agente.
    Args:
        monument_id: UUID exacto del monumento tal como viene del endpoint.
        full_monument_name: Nombre completo y oficial del monumento.
        tag: Categoría del monumento (ej. RELIGIOSO, CIVIL, NATURAL, FORTALEZA, ORNAMENTAL).
    """
    if not monument_id or "especificado" in monument_id.lower():
        return "ERROR: Debes extraer el ID real del monumento desde los resultados de 'get_monuments'."

    consulted = list(runtime.state.get("consulted_ids") or [])
    if monument_id not in consulted:
        consulted.append(monument_id)

    return Command(
        update={
            "monument_id": monument_id,
            "full_monument_name": full_monument_name,
            "tag": tag,
            "consulted_ids": consulted,
            "messages": [
                ToolMessage(
                    content=f"STATE_UPDATED: {full_monument_name} (ID: {monument_id}, categoría: {tag})",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )