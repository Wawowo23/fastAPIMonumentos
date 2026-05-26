import os
import uuid
import json
import httpx

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from output_classes import FinalOrchestratorOutput

# =========================
# CARGAR VARIABLES ENTORNO
# =========================

load_dotenv()

# =========================
# URLS API
# =========================

MONUMENTS_URL = "https://backend-tfg.fly.dev/api/v1/public/monuments"
ROUTES_URL = "https://backend-tfg.fly.dev/api/v1/public/route"

# =========================
# MODELO GROQ + LANGCHAIN
# =========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# TOOLS
# =========================

@tool
def get_monuments() -> str:
    """
    Obtiene la lista completa de monumentos de Martos.
    """

    with httpx.Client(timeout=15.0) as http:

        r = http.get(MONUMENTS_URL)
        r.raise_for_status()

        data = r.json()

        resumen = []

        for m in data:

            sinopsis = next(
                (
                    d["contenido"]
                    for d in m.get("description", [])
                    if d.get("language") == "es"
                    and not d.get("kids")
                    and not d.get("complete")
                ),
                "Sin sinopsis."
            )

            resumen.append(
                f"""
- ID: {m['id']}
- Nombre: {m['name']}
- Categoría: {m['tag']['name']}
- Likes: {m['NLikes']}
- Accesible: {m['accessibility']}
- Sinopsis: {sinopsis}
"""
            )

        return "\n".join(resumen)


@tool
def get_routes() -> str:
    """
    Obtiene rutas turísticas de Martos.
    """

    with httpx.Client(timeout=15.0) as http:

        r = http.get(ROUTES_URL)
        r.raise_for_status()

        data = r.json()

        dificultad_label = {
            0: "Fácil",
            1: "Media",
            2: "Difícil"
        }

        resumen = []

        for route in data:

            tiempo_min = round(
                route.get("estimated_time_seconds", 0) / 60,
                1
            )

            distancia_km = round(
                route.get("total_distance_meters", 0) / 1000,
                2
            )

            diff_val = route.get(
                "difficult",
                route.get("difficulty", 0)
            )

            dif = dificultad_label.get(diff_val, str(diff_val))

            activa = "Sí" if route.get("isActive", False) else "No"

            valoracion = route.get("average_score", 0.0)

            monumentos = [
                f"{m.get('name')} (ID: {m.get('id')})"
                for m in route.get("monuments", [])
                if isinstance(m, dict)
            ]

            resumen.append(
                f"""
- ID: {route['id']}
- Nombre: {route['name']}
- Dificultad: {dif}
- Valoración: {valoracion}/5
- Activa: {activa}
- Distancia: {distancia_km} km
- Tiempo estimado: {tiempo_min} min
- Descripción: {route['description']}
- Monumentos:
{', '.join(monumentos) if monumentos else 'Ninguno'}
"""
            )

        return "\n".join(resumen)


@tool
def get_monument_detail(monument_id: str) -> str:
    """
    Obtiene detalle completo de un monumento.
    """

    with httpx.Client(timeout=15.0) as http:

        r = http.get(MONUMENTS_URL)
        r.raise_for_status()

        data = r.json()

        monument = next(
            (m for m in data if m["id"] == monument_id),
            None
        )

        if not monument:
            return f"No se encontró monumento con ID {monument_id}"

        return json.dumps(monument, ensure_ascii=False)


# =========================
# REGISTRAR TOOLS
# =========================

tools = [
    get_monuments,
    get_routes,
    get_monument_detail
]

# =========================
# BIND TOOLS
# =========================

llm_with_tools = llm.bind_tools(tools)

# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
Eres el asistente oficial de ESCUCHA TU HISTORIA, guía digital del patrimonio cultural de Martos.

REGLA ABSOLUTA:
SIEMPRE usa las herramientas antes de responder.
NUNCA inventes datos.

CUÁNDO USAR CADA HERRAMIENTA:

- Pregunta sobre monumentos:
usa get_monuments

- Pregunta sobre rutas:
usa get_routes

- Pregunta sobre un monumento concreto:
usa get_monuments para obtener el ID
y luego get_monument_detail

CÓMO RESPONDER:

- Responde en español
- Sé amigable y directo
- Usa viñetas e iconos
- Incluye TODOS los datos relevantes
- NUNCA inventes información

Solo rechaza preguntas totalmente ajenas
(recetas, política, deportes, etc).
"""

# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="Escucha Tu Historia — Agente de Monumentos"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST MODEL
# =========================

class ConsultaRequest(BaseModel):

    consulta: str

    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    idioma: Optional[str] = "español"

# =========================
# HEALTHCHECK
# =========================

@app.get("/health")
def health_check():

    return {
        "status": "ok"
    }

# =========================
# ENDPOINT PRINCIPAL
# =========================

@app.post(
    "/consultar",
    response_model=FinalOrchestratorOutput
)
def consultar(body: ConsultaRequest):

    try:

        consulta_con_idioma = (
            f"{body.consulta}\n\n(Responde en {body.idioma})"
            if body.idioma
            and body.idioma.lower() != "español"
            else body.consulta
        )

        messages = [

            SystemMessage(content=SYSTEM_PROMPT),

            HumanMessage(content=consulta_con_idioma)
        ]

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        last_tool_result = None

        while response.tool_calls:

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call["args"]

                selected_tool = next(
                    t for t in tools
                    if t.name == tool_name
                )

                tool_result = selected_tool.invoke(tool_args)

                last_tool_result = tool_result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                })

            response = llm_with_tools.invoke(messages)

            messages.append(response)

        return FinalOrchestratorOutput(
            analisis_final=response.content,
            subagent_json=last_tool_result
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )