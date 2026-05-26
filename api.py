import uuid
import json
import httpx
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from output_classes import FinalOrchestratorOutput
from langsmith import traceable
import os
from langsmith import traceable
from langsmith.wrappers import wrap_openai
load_dotenv()

MONUMENTS_URL = "https://backend-tfg.fly.dev/api/v1/public/monuments"
ROUTES_URL = "https://backend-tfg.fly.dev/api/v1/public/route"

client = Groq()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_monuments",
            "description": "Obtiene la lista completa de monumentos de Martos con IDs, nombres, categorías, likes y accesibilidad.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_routes",
            "description": "Obtiene la lista completa de rutas turísticas de Martos con distancia, tiempo, dificultad y monumentos incluidos.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monument_detail",
            "description": "Obtiene el detalle completo de un monumento: descripciones, imágenes, audios y coordenadas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monument_id": {
                        "type": "string",
                        "description": "UUID exacto del monumento obtenido de get_monuments"
                    }
                },
                "required": ["monument_id"],
            },
        },
    },
]


def execute_tool(tool_name: str, tool_args: dict) -> str:
    try:
        with httpx.Client(timeout=15.0) as http:
            if tool_name == "get_monuments":
                r = http.get(MONUMENTS_URL)
                r.raise_for_status()
                data = r.json()
                resumen = []
                for m in data:
                    sinopsis = next(
                        (d["contenido"] for d in m.get("description", [])
                         if d.get("language") == "es" and not d.get("kids") and not d.get("complete")),
                        "Sin sinopsis."
                    )
                    resumen.append(
                        f"- ID: {m['id']} | Nombre: {m['name']} | Categoría: {m['tag']['name']} | "
                        f"Likes: {m['NLikes']} | Accesible: {m['accessibility']}\n"
                        f"  Sinopsis: {sinopsis}"
                    )
                return "MONUMENTOS DISPONIBLES:\n" + "\n".join(resumen)

            elif tool_name == "get_routes":
                r = http.get(ROUTES_URL)
                r.raise_for_status()
                data = r.json()
                dificultad_label = {0: "Fácil", 1: "Media", 2: "Difícil"}
                resumen = []
                for route in data:
                    tiempo_min = round(route.get("estimated_time_seconds", 0) / 60, 1)
                    distancia_km = round(route.get("total_distance_meters", 0) / 1000, 2)
                    diff_val = route.get("difficult", route.get("difficulty", 0))
                    dif = dificultad_label.get(diff_val, str(diff_val))
                    activa = "Sí" if route.get("isActive", False) else "No"
                    valoracion = route.get("average_score", 0.0)
                    monumentos = [
                        f"{m.get('name')} (ID: {m.get('id')})"
                        for m in route.get("monuments", [])
                        if isinstance(m, dict)
                    ]
                    resumen.append(
                        f"- ID: {route['id']} | Nombre: {route['name']} | Dificultad: {dif} | "
                        f"Valoración: {valoracion}/5 | Activa: {activa}\n"
                        f"  Distancia: {distancia_km} km | Tiempo estimado: {tiempo_min} min\n"
                        f"  Descripción: {route['description']}\n"
                        f"  Monumentos: {', '.join(monumentos) if monumentos else 'Ninguno'}"
                    )
                return "RUTAS DISPONIBLES:\n" + "\n".join(resumen)

            elif tool_name == "get_monument_detail":
                monument_id = tool_args.get("monument_id")
                r = http.get(MONUMENTS_URL)
                r.raise_for_status()
                data = r.json()
                monument = next((m for m in data if m["id"] == monument_id), None)
                if not monument:
                    return f"No se encontró monumento con ID {monument_id}"
                return json.dumps(monument, ensure_ascii=False)

    except httpx.HTTPStatusError as e:
        return f"Error HTTP en {tool_name}: {e.response.status_code}"
    except Exception as e:
        return f"Error en {tool_name}: {str(e)}"


SYSTEM_PROMPT = """Eres el asistente oficial de ESCUCHA TU HISTORIA, guía digital del patrimonio cultural de Martos.

REGLA ABSOLUTA: SIEMPRE usa las herramientas antes de responder. NUNCA inventes datos.

CUÁNDO USAR CADA HERRAMIENTA:
- Pregunta sobre monumentos (lista, likes, categorías, accesibilidad...) → get_monuments
- Pregunta sobre rutas (rutas disponibles, dificultad, distancia...) → get_routes
- Pregunta sobre un monumento concreto → get_monuments para obtener el ID → get_monument_detail con ese ID

CÓMO RESPONDER:
- Responde en español, de forma amigable y directa
- Incluye TODOS los datos relevantes que devuelva la herramienta
- Usa viñetas, negritas e iconos para que sea legible
- NUNCA digas "aquí te dejo la lista" sin escribir la lista completa a continuación
- NUNCA expliques tu proceso técnico al usuario

Solo rechaza con "Lo siento, solo puedo ayudarte con información sobre los monumentos y rutas de Martos 🏛️" si preguntan algo completamente ajeno (recetas, deportes, política, etc)."""


app = FastAPI(title="Escucha Tu Historia — Agente de Monumentos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConsultaRequest(BaseModel):
    consulta: str
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idioma: Optional[str] = "español"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@traceable(name="consulta_monumentos")
@app.post("/consultar", response_model=FinalOrchestratorOutput)
def consultar(body: ConsultaRequest):
    consulta_con_idioma = (
        f"{body.consulta}\n\n(Responde en {body.idioma}.)"
        if body.idioma and body.idioma.lower() != "español"
        else body.consulta
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": consulta_con_idioma},
    ]

    last_tool_result = None

    try:
        for _ in range(5):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048,
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                return FinalOrchestratorOutput(
                    analisis_final=msg.content or "",
                    subagent_json=last_tool_result,
                )

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    } for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")
                result = execute_tool(tool_name, tool_args)

                try:
                    last_tool_result = json.loads(result)
                except Exception:
                    last_tool_result = result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return FinalOrchestratorOutput(
            analisis_final="No pude completar la consulta, inténtalo de nuevo.",
            subagent_json=None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))