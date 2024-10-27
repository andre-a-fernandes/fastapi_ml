from typing import List

from pydantic import BaseModel, conlist


class Iris(BaseModel):
    data: List[conlist(float, min_length=4, max_length=4)] # type: ignore