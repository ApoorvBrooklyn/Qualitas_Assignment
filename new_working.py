import streamlit as st
import requests
import os
import json
from PyPDF2 import PdfReader
from groq import Groq
from dotenv import load_dotenv
import time
import ast

# Load environment variables
load_dotenv()

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"An error occurred while reading the PDF: {e}")
    return text

# Function to classify the extracted text using the LLM
def classification_LLM(text):


    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful classification assistant. You understand engineering concepts. You will be given some text which mostly describes a problem. You have to classify the problem according to a list of choices. More than one choice can also be applicable. Return as a array of applicable CHOICES only. Only return the choices that you are very sure about\n\n#CHOICES\n\n2D Measurement: Diameter, thickness, etc.\n\nAnomaly Detection: Scratches, dents, corrosion\n\nPrint Defect: Smudging, misalignment\n\nCounting: Individual components, features\n\n3D Measurement: Volume, surface area\n\nPresence/Absence: Missing components, color deviations\n\nOCR: Optical Character Recognition, Font types and sizes to be recognized, Reading speed and accuracy requirements\n\nCode Reading: Types of codes to read (QR, Barcode)\n\nMismatch Detection: Specific features to compare for mismatches, Component shapes, color mismatches\n\nClassification: Categories of classes to be identified, Features defining each class\n\nAssembly Verification: Checklist of components or features to verify, Sequence of assembly to be followed\n\nColor Verification: Color standards or samples to match\n"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.21,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None,
    )

    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""
    return answer

def obsjsoncreate(json_template,text,ogtext):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You will be given a text snippet. You will also be given a JSON where some of the fields match with the bullet points in the text. I want you return a JSON where only the fields and subproperties mentioned in the text are present. DONT OUTPUT ANYTHING OTHER THAN THE JSON\n"
            },
            {
                "role": "user",
                "content": "JSON:"+str(json_template)+"\nText:"+text
            }
        ],
        temperature=0.21,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    cutjson=""
    for chunk in completion:
        cutjson += chunk.choices[0].delta.content or ""
    
    completion2 = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. Your task is to populate a JSON structure based on information provided in a PDF document and subsequent user responses. Follow these guidelines carefully:\n\n1. JSON Structure:\n You will be given a JSON template with properties and their descriptions.\nYour goal is to fill the \"User Answer\" subproperty for each field based on the information provided.\n\n2. Information Sources:\nPrimary source: Details extracted from the PDF document.\n\n3. Filling the \"User Answer\":\nIf a clear, unambiguous answer is found, fill it in the \"User Answer\" field.\nIf no information is available or the answer is unclear, mark the field as 'TBD' (To Be Determined).\n\n4. Handling Conflicts:\nMark a field as 'CONFLICT' in the following scenarios:\na: Multiple occurrences of the same field in the PDF with different answers.\nb: Multiple, inconsistent answers provided by the user for the same field.\n\n5. Accuracy and Relevance:\nEnsure that the answers are relevant to the field descriptions.\nDo not infer or assume information that is not explicitly stated.\n\n6. Output Format:\nProvide only the valid, properly formatted JSON as output.\nGive the JSON output with the filled fields only.\nEnsure proper nesting, quotation marks, and commas in the JSON structure.\n\n7. Additional Notes:\nPay attention to units of measurement and formats specified in the field descriptions.\nIf a field requires a specific format (e.g., date, number range), ensure the answer adheres to it.\n\nRemember, your role is to accurately capture and classify the information provided, highlighting any inconsistencies or conflicts. Do not output anything other than the requested JSON structure. Your goal is to provide a clear, accurate, and properly formatted JSON output that reflects the information given, including any ambiguities or conflicts encountered.Give the JSON output with the filled fields only. ENSURE THE JSON IS VALID AND PROPERLY FORMATTED. DO NOT OUTPUT ANYTHING OTHER THAN THE JSON."
            },
            {
                "role": "user",
                "content": "JSON: "+cutjson+"\n Text: "+ogtext
            }
        ],
        temperature=0.21,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    answer = ""
    for chunk in completion2:
        answer += chunk.choices[0].delta.content or ""
    return answer

def bizobjjsoncreate(json_template,text):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion2 = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful classification assistant. You understand engineering concepts. You will be given a JSON where there are properties and their descriptions. You need to fill up the JSON subproperty \"User Answer\" from the details given in the text. If no information is available or the answer is unclear or you are not sure, mark the field as 'TBD' (To Be Determined) and mark a field as 'CONFLICT' in the following scenarios:\na: Multiple occurrences of the same field in the PDF with different answers.\nb: Multiple, inconsistent answers provided by the user for the same field.\n\n Give the JSON output with the filled fields only. ENSURE THE JSON IS VALID AND PROPERLY FORMATTED. DO NOT OUTPUT ANYTHING OTHER THAN THE JSON."
            },
            {
                "role": "user",
                "content": "JSON: "+str(json_template)+"\n Text: "+text
            }
        ],
        temperature=0.21,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    answer = ""
    for chunk in completion2:
        answer += chunk.choices[0].delta.content or ""
    return answer

def question_create(json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a JSON where some subproperties labelled \"User Answer\" are marked as \"TBD\”. I want you to create questions that you as an assistant would ask the user in order to fill up the User Answer field. Create questions to fill these fields, considering the following:\n\n1. For 'TBD' fields, ask for the missing information.\n2. Ensure questions are relevant to the field descriptions.\n3. Pay attention to required formats or units of measurement.\n4. Avoid asking about information already present in the JSON.\n\nReturn all the questions for the user in an array. DO NOT OUTPUT ANYTHING OTHER THAN THE QUESTION ARRAY." 
            },
            {
                "role": "user",
                "content": str(json_template)
            }
        ],
        temperature=0.21,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None,
    )

    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""


    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an experienced writer tasked with refining a set of questions. Follow these guidelines:\n\n1. Ignore any questions about uploading images.\n2. Merge questions asking about different aspects of the same topic.\n3. Maintain a professional yet slightly humorous tone.\n4. Ensure questions are clear and concise.\n5. Avoid redundancy and limit the output to a maximum of 15 questions.\n6. Format the questions to elicit precise answers that can be used in a JSON structure.\n\nRETURN AN ARRAY OF THE REFINED QUESTIONS ONLY. DO NOT RETURN ANYTHING ELSE."
            },
            {
                "role": "user",
                "content": answer
            }
        ],
        temperature=0.73,
        max_tokens=2240,
        top_p=1,
        stream=True,
        stop=None,
    )
    final=""
    for chunk in completion:
        final+=chunk.choices[0].delta.content or ""

    return final

def answer_refill(questions,answers,obs_json_template,bizobj_json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You will be given two arrays: questions and answers. Create a question-answer pair array. For example:\n\n#INPUT\nQuestions=['What is the material of the observed object?', 'What are the dimensions of the object?']\nAnswers=['The object appears to be made of stainless steel', '10 cm x 5 cm x 2 cm']\n\n#OUTPUT\n['Question: What is the material of the observed object? Answer: The object appears to be made of stainless steel','Question: What are the dimensions of the object? Answer: 10 cm x 5 cm x 2 cm']. RETURN ONLY THE FINAL ARRAY OF QUESTION-ANSWER PAIRS."
            },
            {
                "role": "user",
                "content": "Question="+str(questions)+"\nAnswer="+str(answers)
            }
        ],
        temperature=0.5,
        max_tokens=4048,
        top_p=1,
        stream=True,
        stop=None,
    )

    qapair = ""
    for chunk in completion:
        qapair += chunk.choices[0].delta.content or ""
    # print(qapair)
    # print(obs_json_template+bizobj_json_template)
    # print("Question Answer:"+str(qapair)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template))
    completion2 = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a question-answer pair array and a two JSON templates. Follow these guidelines:\n\n1. Fill the \"User Answer\" subproperties in the JSONs based on the question-answer pairs.\n\n2. For fields still marked as \"TBD\" after filling, keep them as \"TBD\".\n\n3. If multiple answers conflict for the same field or there is an answer for an already filled field except \"TBD\", mark its  \"User Answer\" subproperty as \"CONFLICT\"\n\n4. Ensure answers are relevant to field descriptions and adhere to specified formats or units.\n\n5. Do not infer or assume information until not explicitly stated.\n\n6. After filling, merge the two JSONs into a single JSON structure and make sure that there is NO RACE CONDITION while merging.Make sure you return the full JSON, without missing any field. \n\n7. Return the complete, filled, and merged JSON.\n\n8. Ensure the final JSON is valid and properly formatted. DO NOT OUTPUT ANYTHING OTHER THAN THE FINAL MERGED JSON."
            },
            {
                "role": "user",
                "content": "Question Answer:"+str(qapair)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template)
            }
        ],
        temperature=1,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    filled_json=""
    for chunk in completion2:
        filled_json+=chunk.choices[0].delta.content or ""
    # print(filled_json)
    return filled_json


def question_create_conflict(json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a JSON where some subproperties labelled \"User Answer\" are marked as \”CONFLICT\”. I want you to create questions that you as an assistant would ask the user in order to fill up the User Answer field. Create questions to fill these fields, considering the following:\n\n1. For ‘CONFLICT’ fields, ask for the correct and precise information.\n2. Ensure questions are relevant to the field descriptions.\n3. Pay attention to required formats or units of measurement.\n4. Avoid asking about information already present in the JSON.\n\nReturn all the questions for the user in an array. DO NOT OUTPUT ANYTHING OTHER THAN THE QUESTION ARRAY."
            },
            {
                "role": "user",
                "content": str(json_template)
            }
        ],
        temperature=0.21,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None,
    )

    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""


    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an experienced writer tasked with refining a set of questions. Follow these guidelines:\n\n1. Ignore any questions about uploading images.\n2. Merge questions asking about different aspects of the same topic.\n3. Maintain a professional yet slightly humorous tone.\n4. Ensure questions are clear and concise.\n5. Avoid redundancy and limit the output to a maximum of 15 questions.\n6. Format the questions to elicit precise answers that can be used in a JSON structure.\n\nRETURN AN ARRAY OF THE REFINED QUESTIONS ONLY. DO NOT RETURN ANYTHING ELSE."
            },
            {
                "role": "user",
                "content": answer
            }
        ],
        temperature=0.73,
        max_tokens=2240,
        top_p=1,
        stream=True,
        stop=None,
    )
    final=""
    for chunk in completion:
        final+=chunk.choices[0].delta.content or ""

    return final

def answer_refill_conflict(questions,answers,obs_json_template,bizobj_json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You will be given two arrays: questions and answers. Create a question-answer pair array. For example:\n\n#INPUT\nQuestions=['What is the material of the observed object?', 'What are the dimensions of the object?']\nAnswers=['The object appears to be made of stainless steel', '10 cm x 5 cm x 2 cm']\n\n#OUTPUT\n['Question: What is the material of the observed object? Answer: The object appears to be made of stainless steel','Question: What are the dimensions of the object? Answer: 10 cm x 5 cm x 2 cm']. RETURN ONLY THE FINAL ARRAY OF QUESTION-ANSWER PAIRS."
            },
            {
                "role": "user",
                "content": "Question="+str(questions)+"\nAnswer="+str(answers)
            }
        ],
        temperature=0.5,
        max_tokens=4048,
        top_p=1,
        stream=True,
        stop=None,
    )

    qapair = ""
    for chunk in completion:
        qapair += chunk.choices[0].delta.content or ""
    # print(qapair)
    # print(obs_json_template+bizobj_json_template)
    # print("Question Answer:"+str(qapair)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template))
    completion2 = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a question-answer pair array and a two JSON templates. Follow these guidelines:\n\n1. Fill the \"User Answer\" subproperties in the JSONs marked as \"CONFLICT\" based on the question-answer pairs.\n\n2. Ensure answers are relevant to field descriptions and adhere to specified formats or units.\n\n3. Do not infer or assume information until not explicitly stated.\n\n4. After filling, merge the two JSONs into a single JSON structure and make sure that there is NO RACE CONDITION while merging.Make sure you return the full JSON, without missing any field. \n\n5. Return the complete, filled, and merged JSON.\n\n6. Ensure the final JSON is valid and properly formatted. DO NOT OUTPUT ANYTHING OTHER THAN THE FINAL MERGED JSON."
            },
            {
                "role": "user",
                "content": "Question Answer:"+str(qapair)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template)
            }
        ],
        temperature=1,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    filled_json=""
    for chunk in completion2:
        filled_json+=chunk.choices[0].delta.content or ""
    # print(filled_json)
    return filled_json


def executive_summary(json_template):


    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Placeholder for writing the summary status
    status_text = st.empty()
    status_text.text("Writing the summary...")

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a professional copyrighter. You will be given a JSON, I want you to create a complete executive summary with headers and subheaders. It should be a structured document. \"User Answer\" are what are the answers you have to focus on. Dont skip any of the Fields in both JSONs. Use the Description to frame the User answer. DONT OUTPUT ANYTHING OTHER THAN THE SUMMARY."
            },
            {
                "role": "user",
                "content": str(json_template)
            }
        ],
        temperature=0.73,
        max_tokens=5610,
        top_p=1,
        stream=True,
        stop=None,
    )
    final_summ=""
    for chunk in completion:
        final_summ+=chunk.choices[0].delta.content or ""
    status_text.text("Summary generation complete!")

    return final_summ

def chunk_data(data, chunk_size=10):
    if isinstance(data, dict):
        # If data is a dictionary, convert it to a list of key-value pairs
        items = list(data.items())
    elif isinstance(data, list):
        items = data
    else:
        raise TypeError("Data must be either a dictionary or a list")
    
    return [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]

def airtable_write(json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Groq inference
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content":  "You are a helpful assistant. You will be given a unstructured JSON. I want you to convert it into a fully structured JSON which will become a structured CSV. The headings of the CSV are to be \\\"Category\\\",\\\"Sub-category\\\",\\\"Description\\\" and \\\"User Answer\\\". So shuffle around the fields accordingly. \nFields marked \"Category\" are to be directly picked as the \"Category\" for the CSV. If there is \"Observation type\", then that becomes the Category.  \nDONT LEAVE ANY FIELD. MAKE SURE ALL FIELDS ARE INCLUDED IN THE RESULT. DONT OUTPUT ANYTHING OTHER THAN THE JSON. ONLY OUTPUT THE JSON.\n"
            },
            {
                "role": "user",
                "content": json_template
            }
        ],
        temperature=0.25,
        max_tokens=8000,
        top_p=1,
        stream=True,
        # response_format={"type": "json_object"},
        stop=None,
    )
    content=""
    for chunk in completion:
        content+=chunk.choices[0].delta.content or ""
    # Get the structured JSON from Groq
    groq_json = json.loads(content)
    with open("groq_json.json", "w") as file:
        json.dump(groq_json, file, indent=4)
    API_KEY = os.getenv("AIRTABLE_KEY")
    BASE_ID = "appGIi65aZ2YxQrmH"
    TABLE_ID = "Table1"
    url = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}'

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    # Chunk the data into batches of 10
    def chunk_data(data, chunk_size=10):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    # Process each chunk and send it to Airtable
    for batch in chunk_data(groq_json):
        # Format the current batch for Airtable API
        airtable_data = {
            "records": [
                {
                    "fields": {
                        "Category": item["Category"],
                        "Sub-category": item["Sub-category"],
                        "Description": item["Description"],
                        "User Answer": item["User Answer"]
                    }
                } for item in batch
            ]
        }
        
        # Make the POST request to add records
        response = requests.post(url, headers=headers, data=json.dumps(airtable_data))
        
        # Check if the request was successful
        if response.status_code == 200:
            print(f"Batch of {len(batch)} records added successfully!")
        else:
            print(f"Failed to add batch. Status code: {response.status_code}, Error: {response.text}")

            
def main():
    st.title("Qualitas Sales Data Collection Chatbot")
    st.caption("Welcome to the Qualitas Bot. First upload a PDF document which should be customer correspondence, detailing some requirements. Also sometimes the Submit button for the questions is a bit sticky. So You might have to click it twice!")

    # Initialize session state variables
    init_session_state()

    # File uploader for the PDF
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file is not None and not st.session_state.file_processed:
        st.write("Processing your document...")
        process_document(uploaded_file)
        # Display the first question immediately after processing the document
        show_question()

    # Simulate chat interaction
    chat_interaction()

def init_session_state():
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "questionnaire_started" not in st.session_state:
        st.session_state.questionnaire_started = False
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "questionnaire_complete" not in st.session_state:
        st.session_state.questionnaire_complete = False
    if "conflict_questions" not in st.session_state:
        st.session_state.conflict_questions = []
    if "initial_answers" not in st.session_state:
        st.session_state.initial_answers = []
    

def process_document(uploaded_file):
    # Simulate file processing (replace with actual logic)
    st.session_state.text = extract_text_from_pdf(uploaded_file)
    st.session_state.classification_result = classification_LLM(st.session_state.text)
    json_path='observationsJSON.json'
    with open(json_path, 'r') as file:
        obs_json_template = json.load(file)
    final_obs_json = obsjsoncreate(obs_json_template, st.session_state.classification_result, st.session_state.text)
    st.session_state.obs = final_obs_json
    json_path='BizObjJSON.json'
    with open(json_path, 'r') as file:
        bizobj_json_template = json.load(file)
    final_bizobj_json = bizobjjsoncreate(bizobj_json_template, st.session_state.text)
    st.session_state.bizobj = final_bizobj_json
    questionobs = question_create(final_obs_json)
    questionbizobj = question_create(final_bizobj_json)
    while True:
        try:
            # Attempt to evaluate the expressions and assign them to session state
            st.session_state.questions = ast.literal_eval(questionbizobj) + ast.literal_eval(questionobs)
            # If successful, break out of the loop
            break
        except Exception as e:
            # Print the error for debugging purposes (optional)
            print(f"An error occurred: {e}")
            # Wait for 1 second before trying again
            time.sleep(1)
            continue
    st.write(st.session_state.questions)
    # Mark file as processed
    st.session_state.file_processed = True
    st.success("Document processed successfully.")

import json

def check_for_conflicts(completed_json):
    # If completed_json is a string, try to parse it as JSON
    if isinstance(completed_json, str):
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            # Groq inference
            completion = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content":  "You are a helpful assistant. You will be given a unstructured JSON. I want you to convert it into a fully structured JSON which will become a structured CSV. The headings of the CSV are to be \\\"Category\\\",\\\"Sub-category\\\",\\\"Description\\\" and \\\"User Answer\\\". So shuffle around the fields accordingly. \nFields marked \"Category\" are to be directly picked as the \"Category\" for the CSV. If there is \"Observation type\", then that becomes the Category.  \nDONT LEAVE ANY FIELD. MAKE SURE ALL FIELDS ARE INCLUDED IN THE RESULT. DONT OUTPUT ANYTHING OTHER THAN THE JSON. ONLY OUTPUT THE JSON.\n"
                    },
                    {
                        "role": "user",
                        "content": completed_json
                    }
                ],
                temperature=0.25,
                max_tokens=8000,
                top_p=1,
                stream=True,
                # response_format={"type": "json_object"},
                stop=None,
            )
            content=""
            for chunk in completion:
                content+=chunk.choices[0].delta.content or ""
            completed_json = json.loads(completed_json)
        except json.JSONDecodeError:
            st.error("The completed JSON is invalid.")
            return False
    
    # Now ensure it's a dictionary before continuing
    if isinstance(completed_json, dict):
        for key, value in completed_json.items():
            if isinstance(value, dict):
                # Recursively check for conflicts in nested dictionaries
                if check_for_conflicts(value):
                    return True
            elif isinstance(value, list):
                # If the value is a list, check each element
                for item in value:
                    if isinstance(item, dict):
                        if check_for_conflicts(item):
                            return True
                    elif item == "CONFLICT":
                        return True
            elif value == "CONFLICT":
                return True
    return False

def chat_interaction():
    # Display chat messages from history in the correct order
    for message in st.session_state.messages[::-1]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the user's input and show the next question if the questionnaire is not complete
        if not st.session_state.questionnaire_complete:
            show_question()
        else:
            st.write("The questionnaire is complete. Thank you for your responses!")

def show_question():
    if st.session_state.current_question_index < len(st.session_state.questions):
        # Display the next question
        with st.chat_message("assistant"):
            st.markdown(st.session_state.questions[st.session_state.current_question_index])

        # Add the question to the chat history
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.questions[st.session_state.current_question_index]})

        # Move to the next question
        st.session_state.current_question_index += 1

    else:
        # Display the answers after completing the questionnaire
        answers = [message["content"] for message in st.session_state.messages if message["role"] == "user"]

        # Fill in the answers in the JSON templates
        completed_json = answer_refill(st.session_state.questions, answers, st.session_state.obs, st.session_state.bizobj)

        # Check if there are any conflicts in the filled JSON
        if check_for_conflicts(completed_json):
            # Create conflict resolution questions
            conflict_questions = question_create_conflict(completed_json)
            
            # Update the questions and reset the index
            st.session_state.questions = ast.literal_eval(conflict_questions)
            st.session_state.current_question_index = 0
            
            # Display a message about conflicts
            st.write("There are conflicts in the data. Please resolve them by answering the following questions.")
            
            # Display the first conflict question
            with st.chat_message("assistant"):
                st.markdown(st.session_state.questions[0])
            
            # Add the question to the chat history
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.questions[0]})
            
            # Move to the next question
            st.session_state.current_question_index += 1

            

        else:
            # Finalize the output if no conflicts
            airtable_write(completed_json)
            exec_summ = executive_summary(completed_json)
            st.write(exec_summ)
            
            # Reset the questionnaire state
            st.session_state.questionnaire_complete = True

if __name__ == "__main__":
    main()