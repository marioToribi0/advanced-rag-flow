from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.state import GraphState
from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.answer_grader import answer_grader, GradeAnswer
from graph.chains.router import RouteQuery, question_router

load_dotenv()

flow = StateGraph(GraphState)

flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(WEBSEARCH, web_search)
flow.add_node(GENERATE, generate)

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    print("---ROUTE QUESTION TO VECTORSTORE---")
    return RETRIEVE

def is_web_search_neccesary(state: GraphState):
    if (state["web_search"]==True):
        print("---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEBSEARCH---")
        return WEBSEARCH
    return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    hallucinations: GradeHallucinations = hallucination_grader.invoke({"documents": documents, "generation": generation})  
    
    if hallucinations.binary_score=="yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        correct_answer: GradeAnswer = answer_grader.invoke({"question": question, "generation": generation})
        if correct_answer.binary_score=="yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        return "not supported"

flow.set_conditional_entry_point(route_question, {RETRIEVE: RETRIEVE, WEBSEARCH:WEBSEARCH})
flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
flow.add_conditional_edges(GRADE_DOCUMENTS, is_web_search_neccesary, [WEBSEARCH, GENERATE])
flow.add_edge(WEBSEARCH, GENERATE)
flow.add_conditional_edges(
    GENERATE, 
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH
    }
)
flow.add_edge(GENERATE, END)

app = flow.compile()


