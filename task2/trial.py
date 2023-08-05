import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# initialize HF LLM
flan_t5 = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 0.5}
)

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(
    template=multi_template,
    input_variables=["questions"]
)

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=flan_t5
)

qs_str = (
    "Who is the Prime Minister of India?"
)

print(llm_chain.run(qs_str))

