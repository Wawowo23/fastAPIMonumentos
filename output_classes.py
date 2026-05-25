from pydantic import BaseModel, Field
from typing import Any, List, Optional


class Tag(BaseModel):
    id: int = Field(description="Identificador del tag.")
    name: str = Field(description="Nombre de la categoría (ej. RELIGIOSO, CIVIL, NATURAL).")
    colorHex: str = Field(description="Color hexadecimal asociado al tag.")
    createdAt: Optional[str] = Field(default=None, description="Fecha de creación del tag.")


class Coordenates(BaseModel):
    lat: float = Field(description="Latitud del monumento.")
    lon: float = Field(description="Longitud del monumento.")


class DescriptionEntry(BaseModel):
    id: int
    name: str = Field(description="Nombre del tipo de descripción (ej. Sinopsis Español, Texto Audio).")
    language: str = Field(description="Código de idioma (es, en).")
    kids: bool = Field(description="Indica si el contenido está adaptado para niños.")
    complete: bool = Field(description="Indica si el contenido es la versión completa.")
    contenido: str = Field(description="Texto de la descripción.")


class MediaItem(BaseModel):
    id: int
    url: str = Field(description="URL del recurso (imagen o audio).")
    createdAt: Optional[str] = Field(default=None, description="Fecha de creación del recurso.")
    lastModified: Optional[str] = Field(default=None, description="Fecha de última modificación.")


class AudioItem(MediaItem):
    language: str = Field(description="Idioma del audio (es, en).")
    kids: bool = Field(description="Indica si el audio está adaptado para niños.")


class MonumentInfo(BaseModel):
    id: str = Field(description="UUID del monumento.")
    name: str = Field(description="Nombre oficial del monumento.")
    tag: Tag = Field(description="Categoría del monumento.")
    coordenates: Coordenates = Field(description="Coordenadas geográficas.")
    accessibility: bool = Field(description="Si el monumento es accesible.")
    maps_url: str = Field(description="Enlace a Google Maps.")
    NLikes: int = Field(description="Número de likes del monumento.")
    description: List[DescriptionEntry] = Field(description="Lista de descripciones en distintos idiomas y formatos.")
    picture: List[MediaItem] = Field(default=[], description="Lista de imágenes del monumento.")
    audio: List[AudioItem] = Field(default=[], description="Lista de audios del monumento.")


class RouteMonumentInfo(BaseModel):
    id: str = Field(description="UUID del monumento dentro de la ruta.")
    name: str = Field(description="Nombre del monumento.")
    coordenates: Coordenates = Field(description="Coordenadas geográficas del monumento.")
    pictures: List[MediaItem] = Field(default=[], description="Lista de imágenes del monumento.")


class RouteInfo(BaseModel):
    id: str = Field(description="UUID de la ruta.")
    name: str = Field(description="Nombre de la ruta.")
    description: str = Field(description="Descripción breve de la ruta.")
    isActive: bool = Field(description="Indica si la ruta está activa o no.")
    difficult: int = Field(description="Dificultad de la ruta: 0 = fácil, 1 = media, 2 = difícil.")
    monuments: List[RouteMonumentInfo] = Field(description="Lista detallada de los monumentos que componen la ruta.")
    tag: Optional[Tag] = Field(default=None, description="Categoría asignada a la ruta.")
    localidad_id: Optional[int] = Field(default=None, description="ID de la localidad de la ruta.")
    average_score: float = Field(description="Valoración media o puntuación de la ruta.")
    total_distance_meters: float = Field(description="Distancia total en metros.")
    estimated_time_seconds: float = Field(description="Tiempo estimado en segundos.")
    created_at: Optional[str] = Field(default=None, description="Fecha de creación de la ruta.")
    last_modified: Optional[str] = Field(default=None, description="Fecha de última modificación.")


class FinalOrchestratorOutput(BaseModel):
    subagent_json: Any = Field(description="JSON original devuelto por el endpoint consultado.")
    analisis_final: str = Field(
        description="Respuesta FINAL y conversacional que leerá el usuario en su pantalla. "
                    "Debe ser directa, amigable y contener los datos solicitados (rutas, monumentos, etc.). "
                    "NUNCA hables en tercera persona ni expliques tus procesos técnicos."
    )