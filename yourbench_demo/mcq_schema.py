
from pydantic import BaseModel, Field

class DataFormat(BaseModel):
    question: str = Field(description="A self-contained exam-style question answerable strictly from the source text")
    choices: list[str] = Field(description="Exactly 4 answer options, plausible and mutually exclusive")
    answer_index: int = Field(description="0-based index of the correct option in choices")
    explanation: str = Field(description="Brief justification citing the source text")
    difficulty: str = Field(description="easy, medium, or hard")
