import asyncio
import os
import sys

from src.graph.workflow import compile_workflow


async def main():
    print("Starting document summarization workflow...")
    
    # Check if the PDF exists
    pdf_path = "pharma research paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find '{pdf_path}' in the current directory.")
        sys.exit(1)

    # Compile the LangGraph workflow
    workflow = compile_workflow()

    # Initial state
    state = {
        "pdf_path": pdf_path,
        "document_id": "test_doc_001",
        "processing_status": "started",
    }

    # Run the graph
    print(f"Processing '{pdf_path}'...")
    try:
        # invoke handles async internally or we can use ainvoke
        final_state = await workflow.ainvoke(state)
        
        status = final_state.get("processing_status")
        print(f"\nWorkflow finished with status: {status}")
        
        if status == "failed":
            print(f"Errors: {final_state.get('errors')}")
        else:
            print("\n" + "="*50)
            print("FINAL EXECUTIVE SUMMARY:")
            print("="*50)
            print(final_state.get("final_summary", "No summary generated."))
            print("="*50)
            
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")


if __name__ == "__main__":
    asyncio.run(main())
