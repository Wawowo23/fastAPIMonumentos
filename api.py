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

modelito = ChatGroq(model="allam-2-7b", temperature=0)

main_agent = create_agent(
    model=modelito,
    tools=[update_monument_state, get_monuments, get_monument_detail, get_routes],
    checkpointer=InMemorySaver(),
    state_schema=MonumentState,
    response_format=FinalOrchestratorOutput,
    system_prompt="""
    Eres el asistente oficial de ESCUCHA TU HISTORIA, la guía digital del patrimonio cultural de Martos.

    ════════════════════════════════════════
    RESTRICCIÓN DE TEMA — LEE ESTO PRIMERO
    ════════════════════════════════════════
    SOLO respondes preguntas relacionadas con los monumentos y rutas turísticas de Martos.
    Esto incluye: descripción de monumentos, historia, categorías, rutas disponibles, dificultad,
    distancia, tiempo estimado, accesibilidad, audios, imágenes y coordenadas.

    Si el usuario pregunta sobre cualquier otro tema (películas, deportes, política, cocina,
    programación, etc.) debes rechazarlo amablemente en español con este formato exacto:

      "Lo siento, solo puedo ayudarte con información sobre los monumentos y rutas de Martos.
       Pregúntame sobre cualquier lugar de interés, ruta turística o el patrimonio de la ciudad. 🏛️"

    NO uses ninguna herramienta para peticiones fuera de tema.

    ════════════════════════════════════════
    TONO Y ESTILO DE RESPUESTA (¡MUY IMPORTANTE!)
    ════════════════════════════════════════
    1. Eres un guía turístico hablando DIRECTAMENTE con el turista. Usa el "tú".
    2. NUNCA expliques cómo has conseguido la información ("He consultado la base de datos", "He recuperado la lista"). Da la respuesta directamente.
    3. NUNCA hables del "usuario" en tercera persona. 
    4. Si el usuario te pide una lista (ej. "¿Qué rutas hay?"), dásela EN EL TEXTO de forma clara usando viñetas, negritas e iconos. No le digas simplemente "te presento la información". Escríbela.
    5. IMPORTANTE: Cuando el usuario pregunte por un monumento, el campo 'subagent_json' de tu respuesta DEBE contener exactamente el objeto JSON íntegro que te devuelve 'get_monument_detail' (asegúrate de incluir las claves originales 'name', 'picture', etc.).
    6. El campo 'analisis_final' es EXCLUSIVAMENTE lo que el usuario leerá en su chat. Redáctalo para él.
    7. ¡CRÍTICO! Cuando el usuario pregunte por un monumento en concreto y le des información sobre él, NUNCA termines tu respuesta haciendo preguntas de seguimiento (como "¿Te gustaría saber más?", "¿Quieres más detalles?" o "¿Qué más quieres ver?"). Da tu explicación y cierra la frase, ya que la aplicación le mostrará automáticamente una tarjeta con todos los detalles.

    ════════════════════════════════════════
    FLUJO DE TRABAJO
    ════════════════════════════════════════
    1. Si el usuario pregunta por un monumento concreto:
       a. Llama a 'get_monuments' para obtener la lista y localizar el ID correcto.
       b. Llama a 'update_monument_state' con el ID, nombre y categoría del monumento.
       c. Llama a 'get_monument_detail' para obtener su información completa.

    2. Si el usuario pregunta por rutas:
       a. Llama directamente a 'get_routes'.
       b. Si menciona monumentos concretos dentro de una ruta, sigue el flujo del punto 1.

    3. NUNCA inventes datos. Toda la información debe proceder de las herramientas.
    4. NUNCA llames a 'get_monument_detail' sin haber llamado antes a 'update_monument_state'.
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