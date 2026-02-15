from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="""
        Answer strictly with 'yes' if the documents are relevant to the question,
        otherwise answer 'no'. Do not output anything else.
        """
    )

llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt ="""
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',system_prompt),
        ('human', 'Retrieved document: {document} User question: {question}')
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
