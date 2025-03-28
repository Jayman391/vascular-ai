from src.preprocess import ingest_and_prepare_vector_store
from env import make_environment_variables
from src.pipelines import *

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    question = args.question    

    make_environment_variables()
    vectorstore = ingest_and_prepare_vector_store()

    rag_chain = rag_pipeline(vectorstore)
    rag_response = rag_chain.invoke(question)
    print(rag_response)

    graphrag_chain = graphrag_pipeline(vectorstore)
    graphrag_response = graphrag_chain.invoke(question)
    print(graphrag_response)

    full_chain = fusion_chain()
    full_response = full_chain.invoke({"rag_ans": rag_response, "graphrag_ans": graphrag_response})
    print(full_response)

    subquestion_generation_chain = generate_subquestions_chain()
    subquestions = subquestion_generation_chain.invoke(question)
    print(subquestions)

    #split subquestions on \n
    subquestions = subquestions.split("\n")
    subquestion_context = ""
    # for each line in subquestions, query vectorstore and get the context
    for line in subquestions:
      subquestion_context += vectorstore.similarity_search(line, k=1)[0].page_content + "\n"

    print(subquestion_context)

    subquestion_chain = subquestions_chain()
    subquestion_response = subquestion_chain.invoke({
        "context": subquestion_context,
        "answer" : full_response,
        "question": subquestions
    })
    print(subquestion_response)

    final_chain = final_fusion_chain()
    final_response = final_chain.invoke({
        "initial_fused_ans": full_response,
        "subq_ans": subquestion_response
    })
    print(final_response)


