from src.preprocess import ingest_and_prepare_vector_store
from src.pipelines import *

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    question = args.question    

    vectorstore = ingest_and_prepare_vector_store()

    graphrag_chain = graphrag_pipeline(vectorstore)
    graphrag_response = graphrag_chain.invoke(question)

    subquestion_generation_chain = generate_subquestions_chain()
    subquestions = subquestion_generation_chain.invoke(question)

    subquestions = subquestions.split("\n")
    subquestion_context = ""
    for line in subquestions:
      subquestion_context += vectorstore.similarity_search(line, k=2)[0].page_content + "\n"

    subquestion_chain = subquestions_chain()
    subquestion_response = subquestion_chain.invoke({
        "subq_context": subquestion_context,
        "initial_ans" : graphrag_response,
        "question": subquestions
    })

    final_chain = final_fusion_chain()
    final_response = final_chain.invoke({
        "initial_ans": graphrag_response,
        "subq_ans": subquestion_response
    })
    print(final_response)


