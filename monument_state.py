from langchain.agents import AgentState
from typing import List, Optional


class MonumentState(AgentState):
    monument_id: Optional[str] = None
    full_monument_name: Optional[str] = None
    tag: Optional[str] = None
    consulted_ids: List[str] = []