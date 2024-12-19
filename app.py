import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from fpdf import FPDF
from io import BytesIO 
import json
import requests
import base64

# Set OpenAI API Key (replace with your own)
os.environ['OPENAI_API_KEY'] = "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"

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
st.set_page_config(page_title="AI Research Assistant", page_icon=":brain:", layout="wide")
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    /* Center title and sub-heading */
    .stTitle {{
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }}
    .stSubheader {{
        text-align: center;
        font-size: 24px;
        margin-top: 10px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app setup
def analyze_document(file):
    """Analyzes a document and provides a summary using GPT-4."""
    try:
        # Use BytesIO to handle the file-like object from Streamlit
        bytes_data = file.getvalue()  # Get the bytes data
        pdf_file = BytesIO(bytes_data) #Create a BytesIO object
        reader = PdfReader(pdf_file)

        text = ''.join([page.extract_text() for page in reader.pages])

        if not text.strip():  # Check if extracted text is empty
            return "The PDF appears to be empty or contains only images/scanned content which is not readable."
        
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        template = """You are an expert summarizer. Summarize the following text:

        {text}
        """
        prompt = ChatPromptTemplate.from_template(template)
        final_prompt = prompt.format(text=text)
        summary = llm.predict(final_prompt)
        return summary
    except Exception as e:
        st.error(f"Error during document analysis: {e}")
        return None

def analyze_dataset_with_agent(file, file_type, user_query):
    """Analyzes a dataset using LangChain agents."""
    try:
        if file_type == "csv":
            df = pd.read_csv(file)
        elif file_type == "xlsx":
            df = pd.read_excel(file)
        elif file_type == "json":
            df = pd.read_json(file)
        else:
            raise ValueError("Unsupported file type.")

        temp_file = "temp_dataset.csv"
        df.to_csv(temp_file, index=False)

        llm = ChatOpenAI(model="gpt-4", temperature=0)
        tools = [Tool(name="Analyze Dataset", func=lambda x: x, description="Use this to answer questions about the data in the CSV.")]
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        insights = agent.run(f"Here is the data: {df.to_string()}\nUser query: {user_query}")

        os.remove(temp_file)
        return insights
    except Exception as e:
        st.error(f"Error during dataset analysis: {e}")
        return None

def export_summary_to_pdf(summary, output_path="summary.pdf"):
    """Exports a summary to a PDF file."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary)
        pdf.output(output_path)
        return True
    except Exception as e:
        st.error(f"Error exporting to PDF: {e}")
        return False
    
def search_document(file, keyword):
    """Searches for a keyword in a document."""
    try:
        reader = PdfReader(file)
        text = ''.join([page.extract_text() for page in reader.pages])
        occurrences = [line.strip() for line in text.split('\n') if keyword.lower() in line.lower()]
        return occurrences
    except Exception as e:
        st.error(f"Error during keyword search: {e}")
        return None

def main():
    st.title("AI Research Assistant :brain:")
    st.subheader("Analyze Documents 📄, Datasets 📊, and Real-Time Data 🌐 with GPT-4")

    # Set background image by providing the path to your image
    image_base64 = get_base64_image("img1.jpg")  # Replace with your image path
    set_background(image_base64)

    uploaded_file = st.file_uploader("Upload a document or dataset", type=["csv", "pdf", "xlsx", "json"])
    user_query = st.text_input("Ask a question:", placeholder="Enter your question here...")
    keyword = st.text_input("Search for a keyword:", placeholder="Enter keyword to search...")

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "pdf":
            st.write("### Document Analysis")
            summary = analyze_document(uploaded_file)
            if summary:
                st.write("**Summary:**")
                st.markdown(f"<p style='text-align: center; font-size: 16px;'>{summary}</p>", unsafe_allow_html=True)

                if user_query:
                    st.write("**Answer to your question:**")
                    try:
                        doc = Document(page_content=summary)  # Correct Document creation
                        docs = [doc]
                        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                        docs = text_splitter.split_documents(docs)
                        chain = load_qa_chain(ChatOpenAI(temperature=0, model="gpt-4"), chain_type="stuff")
                        answer = chain.run(input_documents=docs, question=user_query)
                        st.markdown(f"<p style='text-align: center; font-size: 18px;'><strong>{answer}</strong></p>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during question answering: {e}")

                if keyword:
                    st.write(f"**Occurrences of '{keyword}':**")
                    occurrences = search_document(uploaded_file, keyword)
                    if occurrences:
                        for occurrence in occurrences:
                            st.markdown(f"<p style='text-align: center; font-size: 16px;'>{occurrence}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='text-align: center;'>No occurrences found.</p>", unsafe_allow_html=True)
                if st.button("Export Summary as PDF"):
                    if export_summary_to_pdf(summary):
                        st.success("Summary exported to summary.pdf")

        elif file_type in ["csv", "xlsx", "json"]:
            st.write("### Dataset Analysis")
            if user_query:
                insights = analyze_dataset_with_agent(uploaded_file, file_type, user_query)
                if insights:
                    st.write("**Answer to your question:**")
                    st.write(insights)
            else:
                st.warning("Please enter a question to analyze the dataset.")
        else:
            st.error("Unsupported file type. Please upload a PDF, CSV, Excel, or JSON file.")

if __name__ == "__main__":
    main()
