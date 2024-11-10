from dotenv import load_dotenv
from graph.graph import app

if __name__ == "__main__":
    load_dotenv()
    app.get_graph().draw_mermaid_png(output_file_path="advanced_rag_flow.png")