import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate

# Load env
load_dotenv()


# Model
model = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
    task="text-generation",
    
)

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Prompt template
template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="""
    Please explain the research paper "{paper_input}" 
    in a {style_input} style with {length_input} detail.
    """
)

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    
    st.write(result if isinstance(result, str) else result.content)
