import os
from typing import Literal, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.load_tools import load_tools
from langsmith import Client

load_dotenv()

client = Client()

with open("./sql_pass.txt") as f:
    passSQL = f.read().strip()

db_uri = f"mysql+mysqlconnector://root:{passSQL}@127.0.0.1/monumentos"
db = SQLDatabase.from_uri(db_uri)

class MensajeHistorial(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AnalizarRequest(BaseModel):
    consulta: str
    chat_history: Optional[List[MensajeHistorial]] = []
    idioma: Optional[str] = "español"

class ConsultaAnalizada(BaseModel):
    """Esquema para la respuesta final."""
    lugares_consultados: List[str]
    respuesta: str

app = FastAPI(title="Agente de Monumentos API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

llm_base = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=1,
)


langsmith_client = Client()

extra_tools = load_tools(["wikipedia"], llm=llm_base)


agent_executor = create_sql_agent(
    llm=llm_base,
    db=db,
    agent_type="tool-calling",
    verbose=True,
    extra_tools=extra_tools,
)

@app.get("/health")
def health_check():
    return {"status": "ok", "db_connected": db is not None}

@app.post("/analizar", response_model=ConsultaAnalizada)
def analizar_consulta(body: AnalizarRequest):
    try:
        
        prompt_template = client.pull_prompt("escucha-tu_historia")

        
        historial_list = [
            (m.role, m.content) for m in body.chat_history
        ] if body.chat_history else []

        
        resultado_agente = agent_executor.invoke({
            "prompt_template": prompt_template,
            "input": body.consulta,
            "chat_history": historial_list,
            "idioma": body.idioma
        })
        
        llm_estructurado = llm_base.with_structured_output(ConsultaAnalizada)
        
        prompt_formateo = (
            f"Toma la siguiente respuesta y asegúrate de que esté en {body.idioma} "
            f"y que las listas tengan saltos de línea (\n) reales: {resultado_agente['output']}"
        )
        
        respuesta_final = llm_estructurado.invoke(prompt_formateo)

        return respuesta_final

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))