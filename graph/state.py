from typing import List, TypedDict

class GraphState(TypedDict):
    """Represents the state of a graph.
    Attributes:
        question: question
        generation: LLM generation for the question.
        web_search: whether to add search or not
        documents: list of retrieved documents for the question.
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]
