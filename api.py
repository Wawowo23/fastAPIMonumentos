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
from langchain_ollama import ChatOllama


from agent_tools import (
    update_monument_state,
    get_monuments,
    get_monument_detail,
    get_routes,
)

load_dotenv()

modelito = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

main_agent = create_agent(
    model=modelito,
    tools=[update_monument_state, get_monuments, get_monument_detail, get_routes],
    checkpointer=InMemorySaver(),
    state_schema=MonumentState,
    response_format=FinalOrchestratorOutput,
    system_prompt="""
    Eres el asistente oficial de ESCUCHA TU HISTORIA, guía digital del patrimonio cultural de Martos.

    IMPORTANT: Antes de responder CUALQUIER pregunta sobre monumentos o rutas, SIEMPRE debes usar las herramientas disponibles para obtener datos reales. NUNCA respondas de memoria.

    CUÁNDO USAR HERRAMIENTAS:
    - Pregunta sobre rutas → llama a get_routes INMEDIATAMENTE
    - Pregunta sobre un monumento → llama a get_monuments, luego update_monument_state, luego get_monument_detail
    - Pregunta general sobre qué hay en Martos → llama a get_monuments

    SOLO rechaza con el mensaje de "Lo siento..." si preguntan sobre temas que NO sean monumentos, rutas o patrimonio de Martos (por ejemplo: recetas, deportes, política, etc).

    "¿Qué rutas hay?" → ES una pregunta válida → USA get_routes
    "¿Qué monumentos hay?" → ES una pregunta válida → USA get_monuments

    CÓMO RELLENAR TU RESPUESTA FINAL:
    - analisis_final: respuesta amigable para el usuario en español
    - subagent_json: JSON raw devuelto por la herramienta (null si no usaste ninguna)
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