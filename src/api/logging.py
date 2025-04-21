from langsmith import Client

def add_response( query: str,response: str,rating: int):  

    client = Client()
    dataset_name = "vascular-ai"

    try:
        client.create_example(
            dataset_name=dataset_name,
            inputs = {"query": query},
            outputs = {"response": response},
            metadata={"rating": rating}
        )
        return True
    
    except Exception as e:
        print(f"Error adding response to LangSmith: {e}")
        return False

