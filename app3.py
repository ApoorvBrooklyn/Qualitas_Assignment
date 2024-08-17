import streamlit as st
import json
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import csv
from PyPDF2 import PdfReader
import re

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the JSON schema
with open('overall_schema.json', 'r') as f:
    original_schema = json.load(f)

# Create a new schema to store filled fields
filled_schema = {category: {field: "" for field in fields} for category, fields in original_schema.items()}

def get_llm_response(prompt, schema, classification, asked_questions):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    full_prompt = f"""
    You are a friendly and professional sales chatbot. Your goal is to gather comprehensive information about a customer's requirements for their business needs. Engage in a natural conversation, asking relevant questions to cover all aspects of the schema and the classification results. Please ensure that you gather all necessary information to fill in the entire schema. If any field is left blank or incomplete, ask a follow-up question to clarify or obtain the missing information. Your objective is to collect complete and accurate data to fill in every field of the schema.

    Current schema state:
    {json.dumps(schema, indent=2)}

    Classification results:
    {classification}

    Asked questions:
    {asked_questions}

    Human: {prompt}

    Assistant: Based on the current context, schema state, classification results, and previously asked questions, provide an appropriate response or ask a relevant question about an aspect of the schema that hasn't been covered yet. Do not repeat questions that have already been asked. If all schema fields have been addressed, summarize the information gathered and ask if there's anything else the customer would like to add or modify.
    """
    
    response = model.invoke(full_prompt)
    return response.content

def extract_information(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    prompt = f"""
    Given the following input, extract relevant information to fill the JSON schema. Don't invent information not present in the input. If a field can't be filled based on the given information, leave it empty.

    Text: {text}

    JSON Schema:
    {json.dumps(original_schema, indent=2)}

    Extracted Information:
    Please provide the extracted information in a structured format, using the category and field names from the schema. For example:
    Category: Field Name: Extracted Value
    If a field has no extractable information, omit it.
    """
    
    response = model.invoke(prompt)
    
    try:
        extracted_info = parse_structured_response(response.content)
        if extracted_info:
            update_schema_with_extracted_info(extracted_info)
        else:
            st.warning("No structured information could be extracted. Please provide more details.")
    except Exception as e:
        st.error(f"An error occurred while extracting information: {str(e)}")
        st.error("Response content: " + response.content[:500] + "...")  # Display first 500 characters of the response

def parse_structured_response(content):
    extracted_info = {}
    current_category = None
    
    for line in content.split('\n'):
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()
                
                if key in original_schema:
                    current_category = key
                    extracted_info[current_category] = {}
                elif current_category and key in original_schema[current_category]:
                    extracted_info[current_category][key] = value
    
    return extracted_info


def manual_extraction(content):
    extracted_info = {}
    lines = content.split('\n')
    current_category = None
    
    for line in lines:
        line = line.strip()
        if line.endswith(':'):
            current_category = line[:-1]
            extracted_info[current_category] = {}
        elif ':' in line and current_category:
            key, value = line.split(':', 1)
            extracted_info[current_category][key.strip()] = value.strip()
    
    return extracted_info if extracted_info else None

def update_schema_with_extracted_info(extracted_info):
    for category, category_data in extracted_info.items():
        if category in filled_schema:
            for key, value in category_data.items():
                if key in filled_schema[category] and value:
                    filled_schema[category][key] = value
    
    st.write("Updated Schema:")
    st.json(filled_schema)

def save_schema_to_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"schema_data_{timestamp}.csv"
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Field", "Value"])
        for category, fields in filled_schema.items():
            for field, value in fields.items():
                writer.writerow([category, field, value])
    return filename

def get_next_question(asked_questions):
    for category in filled_schema:
        for key, value in filled_schema[category].items():
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
    You are a helpful classification assistant. You understand all engineering concepts. You will be given some input which can be in the form of text or pdf which mostly describes a problem. You have to classify the problem according to a list of choices. More than one choice can also be applicable. Return as a list of applicable CHOICES only. Only return the choices that you are very sure about. Following are the list of choices.

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
    return response.content

def main():
    st.set_page_config("Sales Chatbot")
    st.header("Interactive Sales Chatbot 💼")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your sales assistant. I'd like to understand your business needs better. Could you start by telling me about your company and what brings you here today?"})

    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions = set()

    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False

    if "classification_result" not in st.session_state:
        st.session_state.classification_result = ""

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None and not st.session_state.classification_result:
        st.write("Processing your document...")

        with st.spinner('Extracting text and classifying...'):
            text = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.session_state.classification_result = classification_LLM(text)
                
                st.subheader("Classification Result:")
                st.write(st.session_state.classification_result)

                extract_information(text)
            else:
                st.error("No text found in the uploaded PDF.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.conversation_ended:
        prompt = st.chat_input("Your response")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if any(phrase in prompt.lower() for phrase in ["i am done", "all information is given", "that's all", "end conversation"]):
                st.session_state.conversation_ended = True
                response = "Thank you for providing all the information. I'll summarize what we've discussed and save the details. Is there anything else you'd like to add before we conclude?"
            else:
                response = get_llm_response(
                    prompt=prompt, 
                    schema=json.dumps(filled_schema, indent=2),
                    classification=st.session_state.classification_result,
                    asked_questions=", ".join(st.session_state.asked_questions)
                )

            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            extract_information(prompt)
            extract_information(response)

            for category in filled_schema:
                for key in filled_schema[category]:
                    if key.lower() in response.lower():
                        st.session_state.asked_questions.add(key)

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
            st.json(filled_schema)
            st.write("Current Schema State:")
            st.json(filled_schema)

if __name__ == "__main__":
    main()