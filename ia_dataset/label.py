from dataclasses import dataclass
from typing import Optional, Union

from .base import BaseModel


@dataclass
class MLLabel(BaseModel):
    id: Optional[Union[int, str]]
    name: str
    colour: Optional[str] = None
    parent_id: Optional[Union[int, str]] = None