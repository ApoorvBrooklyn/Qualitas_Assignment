import streamlit as st
import json
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import tabula

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the JSON schema
schema = {
    "Metadata": {
        "Company Name": "",
        "Contact Person": ""
    },
    "Requirements": {
        "Problem Statement": "",
        "Current Inspection Methods": "",
        "Challenges with Current System": "",
        "Desired Improvements": "",
        "Key Performance Indicators (KPIs)": "",
        "Return on Investment (ROI) Criteria": "",
        "Future Scalability Needs": "",
        "Integration with Existing Systems": "",
        "Compliance Requirements": "",
        "Environmental Impact Considerations": "",
        "Scalability and Future Expansion": "",
        "Budget Constraints": ""
    }
}

def get_llm_chain():
    prompt_template = """
    You are a friendly and professional sales chatbot. Your goal is to gather comprehensive information about a customer's requirements for their business needs. Engage in a natural conversation, asking relevant questions to cover all aspects of the schema.

    Current conversation context:
    {history}

    Current schema state:
    {schema}

    Asked questions:
    {asked_questions}

    Processed document content:
    {processed_content}

    Human: {input}

    Assistant: Based on the current context, schema state, previously asked questions, and processed document content, provide an appropriate response or ask a relevant question about an aspect of the schema that hasn't been covered yet. Focus on gathering information for unfilled fields in the schema. Do not repeat questions that have already been asked. If all schema fields have been addressed, summarize the information gathered and ask if there's anything else the customer would like to add or modify.
    """
    prompt = PromptTemplate(
        input_variables=["history", "schema", "asked_questions", "processed_content", "input"],
        template=prompt_template
    )
    memory = ConversationBufferMemory(input_key="input", memory_key="history")
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(
        llm=model,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    return chain

def extract_information(text):
    global schema
    prompt = f"""
    Given the following text, extract relevant information to fill the JSON schema. Don't invent information not present in the text. If a field can't be filled based on the given information, leave it empty.

    Text: {text}

    JSON Schema:
    {json.dumps(schema, indent=2)}

    Extracted Information (in JSON format):
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    response = model.invoke(prompt)
    
    try:
        extracted_info = json.loads(response.content)
        for category in extracted_info:
            for key, value in extracted_info[category].items():
                if value and not schema[category][key]:
                    schema[category][key] = value
        return True
    except json.JSONDecodeError:
        st.warning("Failed to extract structured information. Attempting manual extraction.")
        return manual_extraction(response.content)
    except Exception as e:
        st.error(f"An error occurred during extraction: {str(e)}")
        return False

def manual_extraction(content):
    extracted = False
    for category in schema:
        for key in schema[category]:
            if key.lower() in content.lower():
                value_start = content.lower().index(key.lower()) + len(key)
                value_end = content.find('\n', value_start)
                if value_end == -1:
                    value_end = len(content)
                value = content[value_start:value_end].strip(': ')
                if value and not schema[category][key]:
                    schema[category][key] = value
                    extracted = True
    return extracted

def save_schema():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"schema_data_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(schema, f, indent=2)
    return filename

def get_next_question(asked_questions):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    unfilled_fields = []
    for category in schema:
        for key, value in schema[category].items():
            if not value and key not in asked_questions:
                unfilled_fields.append(key)
    
    if not unfilled_fields:
        return None
    
    prompt = f"""
    Based on the following unfilled fields in our schema, generate a natural-sounding question to gather information about one of these fields. Choose the most relevant field to ask about next.

    Unfilled fields: {', '.join(unfilled_fields)}

    Previously asked questions: {', '.join(asked_questions)}

    Generate a single, conversational question to ask the user:
    """
    
    response = model.invoke(prompt)
    return response.content.strip()

def get_pdf_text(pdf_docs):
    text = ""
    tables = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Extract tables
        try:
            tables.extend(tabula.read_pdf(pdf, pages='all', multiple_tables=True))
        except Exception as e:
            st.warning(f"Could not extract tables from PDF: {str(e)}")
    
    # Convert tables to text
    for table in tables:
        text += table.to_string() + "\n\n"
    
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def main():
    st.set_page_config(page_title="Sales Chatbot with Document Upload", layout="wide")
    st.header("Interactive Sales Chatbot 💼")

    if "chain" not in st.session_state:
        st.session_state.chain = get_llm_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your sales assistant. I'd like to understand your business needs better. Could you start by telling me about your company and what brings you here today?"})

    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions = set()

    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False

    if "processed_content" not in st.session_state:
        st.session_state.processed_content = ""

    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Upload Document")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process Document"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.processed_content = raw_text
                extract_information(raw_text)
                st.session_state.document_processed = True
                st.success("Document processed and information extracted")

        st.subheader("Current Schema State")
        st.json(schema)

    with col1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if not st.session_state.conversation_ended:
            if st.session_state.document_processed and not st.session_state.asked_questions:
                next_question = get_next_question(st.session_state.asked_questions)
                if next_question:
                    with st.chat_message("assistant"):
                        st.markdown(next_question)
                        st.session_state.messages.append({"role": "assistant", "content": next_question})
                    st.session_state.asked_questions.add(next_question)

            if prompt := st.chat_input("Your response"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                if any(phrase in prompt.lower() for phrase in ["i am done", "all information is given", "that's all", "end conversation"]):
                    st.session_state.conversation_ended = True
                    response = "Thank you for providing all the information. I'll summarize what we've discussed and save the details. Is there anything else you'd like to add before we conclude?"
                else:
                    response = st.session_state.chain.predict(
                        input=prompt,
                        schema=json.dumps(schema, indent=2),
                        asked_questions=", ".join(st.session_state.asked_questions),
                        processed_content=st.session_state.processed_content
                    )

                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                # Extract information from user input and response
                extract_information(prompt)
                extract_information(response)

                # Update asked questions
                for category in schema:
                    for key in schema[category]:
                        if key.lower() in response.lower():
                            st.session_state.asked_questions.add(key)

                # Check if we need to ask a specific question
                next_question = get_next_question(st.session_state.asked_questions)
                if next_question:
                    with st.chat_message("assistant"):
                        st.markdown(next_question)
                        st.session_state.messages.append({"role": "assistant", "content": next_question})
                    st.session_state.asked_questions.add(next_question)

        if st.session_state.conversation_ended:
            if st.button("Save Collected Information"):
                filename = save_schema()
                st.success(f"Information saved to {filename}")

if __name__ == "__main__":
    main()