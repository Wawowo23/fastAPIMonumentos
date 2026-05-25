import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage
from typing import Literal, Optional
from dotenv import load_dotenv
from output_classes import FinalOrchestratorOutput
from monument_state import MonumentState
from langchain_groq import ChatGroq

from agent_tools import (
    update_monument_state,
    get_monuments,
    get_monument_detail,
    get_routes,
)

load_dotenv()

modelito = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

main_agent = create_agent(
    model=modelito,
    tools=[update_monument_state, get_monuments, get_monument_detail, get_routes],
    checkpointer=InMemorySaver(),
    state_schema=MonumentState,
    response_format=FinalOrchestratorOutput,
    system_prompt="""
Eres el asistente oficial de ESCUCHA TU HISTORIA, la guía digital del patrimonio cultural de Martos.

════════════════════════════════════════
RESTRICCIÓN DE TEMA — OBLIGATORIO
════════════════════════════════════════
ÚNICAMENTE respondes preguntas sobre monumentos y rutas turísticas de Martos.
Temas válidos: descripción, historia, categorías, rutas, dificultad, distancia,
tiempo estimado, accesibilidad, audios, imágenes y coordenadas.

Si el usuario pregunta cualquier otra cosa, responde EXACTAMENTE así y no uses ninguna herramienta:
"Lo siento, solo puedo ayudarte con información sobre los monumentos y rutas de Martos.
Pregúntame sobre cualquier lugar de interés, ruta turística o el patrimonio de la ciudad. 🏛️"

════════════════════════════════════════
CÓMO RELLENAR TU RESPUESTA FINAL
════════════════════════════════════════
Tu respuesta siempre tiene DOS campos:

▸ 'analisis_final'
  - Es el texto que el usuario leerá en su pantalla.
  - Escríbelo en segunda persona ("tú"), de forma amigable y directa.
  - Si te piden una lista, escríbela aquí con viñetas, negritas e iconos.
  - NUNCA expliques cómo has obtenido los datos ("he consultado...", "he recuperado...").
  - NUNCA hagas preguntas de seguimiento al final ("¿quieres saber más?", etc.).
  - NUNCA hables del "usuario" en tercera persona.

▸ 'subagent_json'
  - CRÍTICO: Este campo debe contener el JSON RAW que la herramienta te ha devuelto.
  - Es decir: el CONTENIDO de la respuesta de la tool, no su nombre ni sus argumentos.
  - Ejemplo CORRECTO → el objeto JSON con los datos del monumento o las rutas.
  - Ejemplo INCORRECTO → {{"function_name": "get_routes", "args": []}}
  - Si la pregunta no involucra ningún monumento o ruta concretos, pon null.

════════════════════════════════════════
FLUJO DE TRABAJO — SIGUE ESTE ORDEN
════════════════════════════════════════

CASO 1 — El usuario pregunta por un monumento concreto:
  1. Llama a 'get_monuments' → obtén la lista y localiza el ID exacto.
  2. Llama a 'update_monument_state' con el ID, nombre completo y categoría.
  3. Llama a 'get_monument_detail' → obtendrás el JSON completo del monumento.
  4. Pon ese JSON íntegro en 'subagent_json'.
  5. Redacta 'analisis_final' con la información relevante para el turista.

CASO 2 — El usuario pregunta por rutas:
  1. Llama a 'get_routes' → obtendrás el JSON con todas las rutas.
  2. Pon ese JSON íntegro en 'subagent_json'.
  3. Redacta 'analisis_final' listando las rutas de forma clara.
  4. Si menciona un monumento concreto dentro de la ruta, sigue el CASO 1.

REGLAS INAMOVIBLES:
  - NUNCA inventes datos. Todo debe venir de las herramientas.
  - NUNCA llames a 'get_monument_detail' sin haber llamado antes a 'update_monument_state'.
  - 'subagent_json' = datos reales devueltos por la tool, NUNCA metadatos de la llamada.
"""
)


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


@app.post("/consultar", response_model=FinalOrchestratorOutput)
def consultar(body: ConsultaRequest):
    config = {"configurable": {"thread_id": body.thread_id}}

    consulta_con_idioma = (
        f"{body.consulta}\n\n(Responde en {body.idioma}.)"
        if body.idioma and body.idioma.lower() != "español"
        else body.consulta
    )

    try:
        response = main_agent.invoke(
            {"messages": [HumanMessage(content=consulta_con_idioma)]},
            config=config,
        )
    except Exception as e:
        if "INVALID_CHAT_HISTORY" in str(e) or "tool_calls" in str(e):
            config_limpio = {"configurable": {"thread_id": str(uuid.uuid4())}}
            try:
                response = main_agent.invoke(
                    {"messages": [HumanMessage(content=consulta_con_idioma)]},
                    config=config_limpio,
                )
            except Exception as e2:
                raise HTTPException(status_code=500, detail=str(e2))
        else:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        return response["structured_response"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))