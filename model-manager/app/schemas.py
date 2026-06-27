from typing import Optional

from pydantic import BaseModel


class RegisterAdapterRequest(BaseModel):
    id: str
    path: str
    rank: Optional[int] = None
    description: Optional[str] = None


class AdapterView(BaseModel):
    id: str
    base_model: str
    path: str
    rank: Optional[int] = None
    state: str
    pinned: bool
    in_flight: int
    last_used: Optional[float] = None
    description: Optional[str] = None


class ModelsResponse(BaseModel):
    capacity: int
    loaded_count: int
    models: list[AdapterView]


class ErrorDetail(BaseModel):
    message: str
    type: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
