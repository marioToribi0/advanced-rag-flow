from dotenv import load_dotenv
load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    # app.get_graph().draw_mermaid_png(output_file_path="advanced_rag_flow.png")
    
    result = app.invoke(
        input={
            "question": "Que es prompt engineering?"
    })["generation"]
    
    print("\n****\n")
    print(result)