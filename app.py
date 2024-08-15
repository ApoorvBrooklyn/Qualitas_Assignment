import streamlit as st
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import csv
from pypdf import PdfReader

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
    You are a friendly and professional sales chatbot. Your goal is to gather comprehensive information about a customer's requirements for their business needs. Engage in a natural conversation, asking relevant questions to cover all aspects of the schema and the classification results.

    Current conversation context:
    {history}

    Current schema state:
    {schema}

    Classification results:
    {classification}

    Asked questions:
    {asked_questions}

    Human: {input}

    Assistant: Based on the current context, schema state, classification results, and previously asked questions, provide an appropriate response or ask a relevant question about an aspect of the schema that hasn't been covered yet. Do not repeat questions that have already been asked. If all schema fields have been addressed, summarize the information gathered and ask if there's anything else the customer would like to add or modify.
    """

    prompt = PromptTemplate(
        input_variables=["history", "schema", "classification", "asked_questions", "input"],
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
        extracted_info = json.loads(response.text)
        for category in extracted_info:
            for key, value in extracted_info[category].items():
                if value and not schema[category][key]:  # Only update if a value was extracted and the field is empty
                    schema[category][key] = value
    except json.JSONDecodeError:
        st.error("Failed to extract information. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def save_schema_to_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"schema_data_{timestamp}.csv"
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Field", "Value"])
        for category, fields in schema.items():
            for field, value in fields.items():
                writer.writerow([category, field, value])
    return filename

def get_next_question(asked_questions):
    for category in schema:
        for key, value in schema[category].items():
            if not value and key not in asked_questions:
                return f"Could you please provide information about the {key}?"
    return None

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"An error occurred while reading the PDF: {e}")
    return text

def classification_LLM(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.15)
    prompt = f"""
    You are a helpful classification assistant. You understand engineering concepts. You will be given some text which mostly describes a problem. You have to classify the problem according to a list of choices. More than one choice can also be applicable. Return as a list of applicable CHOICES only. Only return the choices that you are very sure about

    CHOICES:
    2D Measurement
    Anomaly Detection
    Print Defect
    Counting
    3D Measurement
    Presence/Absence
    OCR
    Code Reading
    Mismatch Detection
    Classification
    Assembly Verification
    Color Verification

    Text: {text}

    Classification:
    """
    response = model.invoke(prompt)
    return response.text

def main():
    st.set_page_config("Sales Chatbot")
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

    if "classification_result" not in st.session_state:
        st.session_state.classification_result = ""

    # File uploader for the PDF
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None and not st.session_state.classification_result:
        st.write("Processing your document...")

        # Show a spinner while processing
        with st.spinner('Extracting text and classifying...'):
            # Extract text from the PDF
            text = extract_text_from_pdf(uploaded_file)
            
            if text:
                # Classify the extracted text
                st.session_state.classification_result = classification_LLM(text)
                
                # Display the classification result
                st.subheader("Classification Result:")
                st.write(st.session_state.classification_result)
            else:
                st.error("No text found in the uploaded PDF.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.conversation_ended:
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
                    classification=st.session_state.classification_result,
                    asked_questions=", ".join(st.session_state.asked_questions)
                )

            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

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
                st.session_state.asked_questions.add(next_question.split("about the ")[-1].rstrip("?"))

    if st.session_state.conversation_ended:
        if st.button("Save Collected Information"):
            filename = save_schema_to_csv()
            st.success(f"Information saved to {filename}")
            st.json(schema)

if __name__ == "__main__":
    main()