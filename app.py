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

    Human: {input}

    Assistant: Based on the current context, schema state, and previously asked questions, provide an appropriate response or ask a relevant question about an aspect of the schema that hasn't been covered yet. Do not repeat questions that have already been asked. If all schema fields have been addressed, summarize the information gathered and ask if there's anything else the customer would like to add or modify.
    """

    prompt = PromptTemplate(
        input_variables=["history", "schema", "asked_questions", "input"],
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
        response_content = response.content if hasattr(response, 'content') else str(response)
        extracted_info = json.loads(response_content)
        for category in extracted_info:
            for key, value in extracted_info[category].items():
                if value and not schema[category][key]:  # Only update if a value was extracted and the field is empty
                    schema[category][key] = value
    except json.JSONDecodeError:
        st.error("Failed to extract information. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def save_schema():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"schema_data_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(schema, f, indent=2)
    return filename

def get_next_question(asked_questions):
    for category in schema:
        for key, value in schema[category].items():
            if not value and key not in asked_questions:
                return f"Could you please provide information about the {key}?"
    return None

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
            filename = save_schema()
            st.success(f"Information saved to {filename}")
            st.json(schema)

if __name__ == "__main__":
    main()