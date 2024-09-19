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
                "content": "You are a helpful assistant. You will be given a text snippet. You will also be given a JSON where some of the fields match with the bullet points in the text. I want you return a JSON where only the fields and subproperties mentioned in the text are present.ENSURE THE JSON IS VALID AND PROPERLY FORMATTED. DONT OUTPUT ANYTHING OTHER THAN THE JSON\n"
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
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. Your task is to populate a JSON structure based on information provided in a PDF document and subsequent user responses. Follow these guidelines carefully:\n\n1. JSON Structure:\n You will be given a JSON template with properties and their descriptions.\nYour goal is to fill the \"User Answer\" subproperty for each field based on the information provided.\n\n2. Information Sources:\nPrimary source: Details extracted from the PDF document.\n\n3. Filling the \"User Answer\":\nIf a clear, unambiguous answer is found, fill it in the \"User Answer\" field.\nIf no information is available or the answer is unclear, mark the field as 'TBD.\n\n Mark a field as 'CONFLICT' in the following scenarios:\na: Multiple occurrences of the same field in the PDF with different answers.\nb: Multiple, inconsistent answers provided by the user for the same field.\n\n5. Accuracy and Relevance:\nEnsure that the answers are relevant to the field descriptions.\nDo not infer or assume information until explicitly stated.\n\n6. Output Format:\nProvide only the valid, properly formatted JSON as output.\nGive the JSON output with the filled fields only.\nEnsure proper nesting, quotation marks, and commas in the JSON structure.\n\n7. Also:\nPay attention to units of measurement and formats specified in the field descriptions.\nIf a field requires a specific format (e.g., date, number range), ensure the answer adheres to it.\n\nRemember, your role is to accurately capture and classify the information provided, highlighting any inconsistencies or conflicts. Do not output anything other than the requested JSON structure. Your goal is to provide a clear, accurate, and properly formatted JSON output that reflects the information given, including any ambiguities or conflicts encountered.Give the JSON output with the filled fields only. ENSURE THE JSON IS VALID AND PROPERLY FORMATTED. DO NOT OUTPUT ANYTHING OTHER THAN THE JSON."
            },
            {
                "role": "user",
                "content": "JSON: "+cutjson+"\n Text: "+ogtext
            }
        ],
        temperature=0.23,
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
                "content": "You are a helpful classification assistant. You understand engineering concepts. You will be given a JSON where there are properties and their descriptions. You need to fill up the JSON subproperty \"User Answer\" from the details given in the text. If no information is available or the answer is unclear or you are not sure, mark the field as 'TBD' (To Be Determined) and mark a field as 'CONFLICT' in the following scenario:\nMultiple occurrences of the same field in the text with different answers.\n For Example if the Budget Constraints are mentioned twice or more in the PDF Input with contrasting values then that field should be marked as CONFLICT\n\n Give the JSON output with the filled fields only. Make sure that you consider all categories which are BIZ_OBJ, PROD_VARIANT_INFO, MATERIAL_HANDLING, SOFTWARE, CUSTOMER_DEPENDENCY and ACCEPTANCE. ENSURE THE JSON IS VALID AND PROPERLY FORMATTED. DO NOT OUTPUT ANYTHING OTHER THAN THE JSON. THE OUTPUT JSON SHOULD BE SAME AS THE GIVEN JSON JUST WITH FILLED USER ANSWERS FROM THE TEXT OR MARKED AS 'TBD' OR 'CONFLICT'."
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
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a JSON where some subproperties labelled \"User Answer\" are marked as \"TBD\". I want you to create questions that you as an assistant would ask the user in order to fill up the \"User Answer\" field. Create questions to fill these fields, considering the following:\n\n1. For \"TBD\" fields, ask for the missing information.\n2. Ensure questions are relevant to the field descriptions.\n3. Pay attention to required formats or units of measurement.\n4. Avoid asking questions and information about the fields which are not marked as 'TBD' in the JSON.\n\nReturn all the questions for the user in an array. Make sure that you consider all categories which are BIZ_OBJ, PROD_VARIANT_INFO, MATERIAL_HANDLING, SOFTWARE, CUSTOMER_DEPENDENCY and ACCEPTANCE. DO NOT MISS OUT ON ANY FIELD WITH \"User Answer\" SUB PROPERTY AS \"TBD\".DO NOT OUTPUT ANYTHING OTHER THAN THE QUESTION ARRAY." 
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
                "content": "You are an experienced writer tasked with refining a set of questions. Follow these guidelines:\n\n1. Ignore all questions which requires uploading of images.\n2. Merge two or more questions asking about different aspects of the same topic.\n3. Maintain a professional yet slightly funny tone.\n4. Ensure questions are clear and concise.\n5. AVOID REDUNDANCY and limit the output to a maximum of 15 questions. Make sure that all the questions are asked within these 15 questions. All questions should be included in the least number of questions possible with the maximum being 15 questions.\n6. Format the questions to elicit precise answers that can be stored in a JSON structure.\n\nRETURN AN ARRAY OF THE REFINED QUESTIONS ONLY WITH NO OTHER FIELDS EXCEPT THE QUESTION ITSELF. MAKE SURE THE QUESTION DO NOT EXCEED 15 QUESTIONS. DO NOT RETURN ANYTHING ELSE."
            },
            {
                "role": "user",
                "content": answer
            }
        ],
        temperature=0.23,
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
                "content": "You are a helpful assistant. You will be given two arrays: `questions` and `answers`. Your task is to create a list of question-answer pairs in the format 'Question: [question] Answer: [answer]' for each question and its corresponding answer. For example: Questions = ['What is the material of the observed object?', 'What are the dimensions of the object?'],Answers = ['The object appears to be made of stainless steel', '10 cm x 5 cm x 2 cm']['Question: What is the material of the observed object? Answer: stainless steel', 'Question: What are the dimensions of the object? Answer: 10 cm x 5 cm x 2 cm']. RETURN ONLY THE FINAL ARRAY OF QUESTION-ANSWER PAIRS."
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
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a question-answer pair array and a two JSON templates. Follow these guidelines:\n\n1. Fill the \"User Answer\" subproperties in the JSONs based on the question-answer pairs. There might be a possiblity that the answer or a part of the answer is not relevant to the question but gives information about some other field in the JSON so classify the answers considering all the fields present in the JSON templates not just the fields relevant to the question.\n\n2. For fields still marked as \"TBD\" after filling, keep them as \"TBD\".\n\n3. If multiple answers conflict for the same field or there is an answer for an already filled field except \"TBD\", mark its  \"User Answer\" subproperty as \"CONFLICT\". Suppose in two answers the user gives the problem statement or budget constraint or any other field information twice with contrasting data then mark it as 'CONFLICT'. This situation can also arise when the \"User Answer\" subproperty is already filled in the given JSON input and the answer given by the userr to any of the questions provides conflicting information about the same field then that field \"User Answer\" subproperty should also be marked as 'CONFLICT'\n\n4. Ensure answers are relevant to field descriptions and adhere to specified formats or units.\n\n5. Do not infer or assume information until not explicitly stated.\n\n6. After filling, merge the two JSONs into a single JSON structure and make sure that there is NO RACE CONDITION while merging.\n\n7. Make sure to Return the complete, filled, and merged JSON without missing any field.\n\n8. Ensure the final JSON is valid and properly formatted without any error. Make sure that the final JSON returned has all fields and subproperties as given in the input just with the User Answers filled accordingly. DO NOT OUTPUT ANYTHING OTHER THAN THE FINAL MERGED JSON."
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

def text_refill(text,obs_json_template,bizobj_json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion= client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a text and two JSON templates. Follow these guidelines:\n\n1. Fill the \"User Answer\" subproperties in the JSONs based on the text. Fill the subproperty on the basis of what is written in the text and classify it into the various fields of the JSON.\n\n2. For fields still marked as \"TBD\" after filling, keep them as \"TBD\".\n\n3. If multiple answers conflict for the same field or there is an answer for an already filled field except \"TBD\", mark its  \"User Answer\" subproperty as \"CONFLICT\". Suppose in two answers the user gives the problem statement or budget constraint or any other field information twice with contrasting data then mark it as 'CONFLICT'. This situation can also arise when the \"User Answer\" subproperty is already filled in the given JSON input and the answer given by the user to any of the questions provides conflicting information about the same field then that field \"User Answer\" subproperty should also be marked as 'CONFLICT'\n\n4. Ensure answers are relevant to field descriptions and adhere to specified formats or units.\n\n5. Do not infer or assume information until not explicitly stated.\n\n6. After filling, merge the two JSONs into a single JSON structure and make sure that there is NO RACE CONDITION while merging. Make sure you return the full JSON, without missing any field. \n\n7. Return the complete, filled, and merged JSON.\n\n8. Ensure the final JSON is valid and properly formatted. Make sure that the final JSON returned has all fields and subproperties as given in the input just with the User Answers filled accordingly. DO NOT OUTPUT ANYTHING OTHER THAN THE FINAL MERGED JSON."
            },
            {
                "role": "user",
                "content": "Text:"+str(text)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template)
            }
        ],
        temperature=0.53,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    filled_json=""
    for chunk in completion:
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
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a JSON where some subproperties labelled \"User Answer\" are marked as \"CONFLICT\" or \"TBD\". I want you to create questions that you as an assistant would ask the user in order to fill up the User Answer field. Create questions to fill these fields, considering the following:\n\n1. For 'CONFLICT' or 'TBD' fields, ask for the correct and precise information.\n2. Ensure questions are relevant to the field descriptions.\n3. Pay attention to required formats or units of measurement.\n4. Avoid asking about information already present in the JSON.\n\nReturn all the questions for the user in an array. DO NOT OUTPUT ANYTHING OTHER THAN THE QUESTION ARRAY. RETURN ONLY QUESTIONS DO NOT WRITE ANYTHING LIKE python or any other specifiers."
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
                "content": "You are an experienced writer tasked with refining a set of questions. Follow these guidelines:\n\n1. Ignore any questions which requires uploading images.\n2. Merge questions asking about different aspects of the same fields.\n3. Maintain a professional yet slightly humorous tone.\n4. Ensure questions are clear and concise.\n5. AVOID REDUNDANCY and limit the output to a maximum of 15 questions. Make sure that all the questions are asked within these 15 questions. All questions should be included in the least number of questions possible with the maximum being 15 questions. Do not ask any extra questions.\n6. Format the questions to elicit precise answers that can be used in a JSON structure.\n\nRETURN AN ARRAY OF THE REFINED QUESTIONS ONLY WITH NO OTHER FIELDS EXCEPT THE QUESTION ITSELF. MAKE SURE THE QUESTION DO NOT EXCEED 15 QUESTIONS. DO NOT RETURN ANYTHING ELSE."
            },
            {
                "role": "user",
                "content": answer
            }
        ],
        temperature=0.23,
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
                "content": "You are a helpful assistant. You will be given two arrays: questions and answers. Create a question-answer pair array for each question asked for conflict fields. For example:\n\n#INPUT\nQuestions=['What is the material of the observed object?', 'What are the dimensions of the object?']\nAnswers=['The object appears to be made of stainless steel', '10 cm x 5 cm x 2 cm']\n\n#OUTPUT\n['Question: What is the material of the observed object? Answer: stainless steel','Question: What are the dimensions of the object? Answer: 10 cm x 5 cm x 2 cm']. RETURN ONLY THE FINAL ARRAY OF QUESTION-ANSWER PAIRS."
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
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a question-answer pair array and a two JSON templates. Follow these guidelines:\n\n1. Fill the \"User Answer\" subproperties in the JSONs marked as \"CONFLICT\" or \"TBD\" based on the question-answer pairs.\n\n2. Ensure answers are relevant to field descriptions and adhere to specified formats or units.\n\n3. Do not infer or assume information until not explicitly stated.\n\n4. After filling, merge the two JSONs into a single JSON structure and make sure that there is NO RACE CONDITION while merging.Make sure you return the full JSON, without missing any field. \n\n5. Return the complete, filled, and merged JSON.\n\n6. Ensure the final JSON is valid and properly formatted. Make sure that the final JSON returned has all fields and subproperties as given in the input just with the User Answers filled accordingly. DO NOT OUTPUT ANYTHING OTHER THAN THE FINAL MERGED JSON."
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
        temperature=0.53,
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
                "content": 
                    """You are an AI assistant specializing in JSON data restructuring. Your task is to convert unstructured JSON into a structured JSON array that can be easily transformed into a CSV format. Each object in the output array must have exactly four fields: "Category", "Sub-category", "Description", and "User Answer". Map the input fields as follows: use "Category" or "Observation type" for the "Category" field; "Field Name" or "Sub-Parameters" for the "Sub-category" field; "Description" or "Example" for the "Description" field; and fields explicitly marked as "User Answer" for the "User Answer" field. If multiple instances of the same field type exist, combine them logically or use the most relevant one. If a field doesn't clearly map to one of these four, use your best judgment to place it appropriately. Include all relevant information from the input, ensuring no data is lost. Your output must be a valid, properly formatted JSON array with no missing braces, quotes, or commas. Do not include any explanations, comments, or text outside the JSON structure. If the input is ambiguous or challenging to structure, still attempt to produce a valid JSON output with the available information. Your response should consist exclusively of the restructured JSON array, nothing else, as this output will be directly passed to an Airtable write function.\n """
            },
            {
                "role": "user",
                "content": json_template
            }
        ],
        temperature=0.27,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    content = ""
    for chunk in completion:
        content += chunk.choices[0].delta.content or ""
    
    print("Raw content from Groq:")
    print(content[:1000] + "..." if len(content) > 1000 else content)  # Print first 1000 chars for very long content
    
    # Write raw content to a file for potential debugging
    with open("raw_groq_output.json", "w") as file:
        file.write(content)
    
    # Parse the JSON content
    records = parse_json_content(content)
    
    if not records:
        print("Failed to parse JSON content. Exiting.")
        return

    API_KEY = os.getenv("AIRTABLE_KEY")
    BASE_ID = "appGIi65aZ2YxQrmH"
    TABLE_ID = "Table1"
    url = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}'

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    # Process records and send to Airtable
    process_records(records, url, headers)

    print("Airtable write operation completed.")

def parse_json_content(content):
    try:
        # First, try to parse as a JSON array
        records = json.loads(content)
        if isinstance(records, list):
            return records
        elif isinstance(records, dict):
            return [records]
    except json.JSONDecodeError:
        print("Failed to parse as a complete JSON. Attempting to parse line by line.")
        
        # If that fails, try to parse line by line
        records = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                except json.JSONDecodeError:
                    print(f"Failed to parse line: {line}")
        
        if records:
            return records
    
    print("Failed to parse JSON content.")
    return None

def process_records(records, url, headers, batch_size=10):
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        airtable_data = {
            "records": [
                {
                    "fields": {
                        "Category": item.get("Category", ""),
                        "Sub-category": item.get("Sub-category", ""),
                        "Description": item.get("Description", ""),
                        "User Answer": item.get("User Answer", "")
                    }
                } for item in batch
            ]
        }
        
        send_to_airtable(url, headers, airtable_data)

def send_to_airtable(url, headers, data):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            print(f"Batch of {len(data['records'])} records added successfully!")
            return
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed. Error: {e}")
            if hasattr(e, 'response'):
                print(f"Response content: {e.response.text}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Failed to send batch to Airtable.")

def main():
    st.markdown("""
    <style>
        .stApp {
            background-color: white;
            color: #262730;
        }
        .stTextInput > div > div > input {
            background-color: #f0f2f6;
            color: black;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stMarkdown {
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #1E88E5;
        }
        h2 {
            color: #43A047;
        }
        .stAlert {
            background-color: #E3F2FD;
            color: #1565C0;
        }
        .caption {
            font-size: 0.9em;
            color: #555;
            font-style: italic;
            margin-bottom: 20px;
        }
        /* Ensure all text is visible */
        .stChatMessage, .stChatMessage p {
            color: black !important;
        }
        /* Style for questions */
        .question {
            background-color: #625d5d;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        /* Ensure chat input is visible */
        .stChatInputContainer {
            background-color: #f0f2f6 !important;
        }
        .stChatInputContainer textarea {
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)
        
    st.title("Qualitas Sales Data Collection Chatbot")
    st.markdown("<p class='caption'>Welcome to the Qualitas Bot. First upload a PDF document which should be customer correspondence, detailing some requirements. Also sometimes the Submit button for the questions is a bit sticky. So You might have to click it twice!</p>", unsafe_allow_html=True)
    # Initialize session state variables
    init_session_state()

    # File uploader for the PDF
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file is not None and not st.session_state.file_processed:
        st.write("Processing your document...")
        process_document(uploaded_file)
        # Display the first question immediately after processing the document
        show_question()

    st.sidebar.markdown("## Upload Additional Text or Enter Text")

    # Option to select either upload or write text
    option = st.sidebar.radio("Choose input method:", ("Upload PDF", "Write Text"))

    if option == "Upload PDF":
        pdf_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
        if pdf_file is not None:
            text_content = extract_text_from_pdf(uploaded_file)
            if st.sidebar.button("Process PDF"):
                process_additional_text(text_content)

    elif option == "Write Text":
        text_content = st.sidebar.text_area("Enter your text here", value=st.session_state.sidebar_text_input)
        if st.sidebar.button("Process Written Text"):
            process_additional_text(text_content)
            # Clear the text area
            st.session_state.sidebar_text_input = ""
            # To ensure the text area reflects the cleared state

    # Simulate chat interaction
    chat_interaction()

def process_additional_text(text_content):
    st.write("Processing additional text...")
    
    # Call text_refill function
    updated_json = text_refill(text_content, st.session_state.obs, st.session_state.bizobj)
    
    # Update session state with new JSON
    st.session_state.obs = updated_json
    st.session_state.bizobj = updated_json
    if check_for_conflicts(updated_json):
        # Conflicts found, start conflict resolution
        conflict_questions = question_create_conflict(updated_json)
        st.session_state.questions = ast.literal_eval(conflict_questions)
        st.session_state.current_question_index = 0
        st.session_state.conflict_resolution_mode = True
        st.session_state.messages = []
        st.empty()
        st.write("Conflicts detected. Please answer the following questions to resolve them.")
        st.write(st.session_state.questions)
        st.success("Additional text processed. New questions generated.")
        show_question()
    else:
        finalize_questionnaire(updated_json)
    
    

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
    if "conflict_resolution_mode" not in st.session_state:
        st.session_state.conflict_resolution_mode = False
    if "sidebar_text_input" not in st.session_state:
        st.session_state.sidebar_text_input = ""

    
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


def check_for_conflicts(completed_json: str) -> bool:
    # Check if 'CONFLICT' is present in the string (case-insensitive)
    if 'CONFLICT' in completed_json.upper():
        return True
    elif 'TBD' in completed_json.upper():
        return True
    else:
        return False


def chat_interaction():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is the answer?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not st.session_state.questionnaire_complete:
            show_question()
        else:
            st.empty()
            st.write("The questionnaire is complete. Thank you for your responses!")


def show_question():
    # Display the next question if available
    if st.session_state.current_question_index == 0:
        question = st.session_state.questions[st.session_state.current_question_index]
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.questions[st.session_state.current_question_index]})
        st.session_state.current_question_index += 1
    elif st.session_state.current_question_index < len(st.session_state.questions):
        question = st.session_state.questions[st.session_state.current_question_index]
        with st.chat_message("assistant"):
            st.markdown(f'<div class="question">{question}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.questions[st.session_state.current_question_index]})
        st.session_state.current_question_index += 1
    else:
        # Once all questions are answered
        
        answers = [message["content"] for message in st.session_state.messages if message["role"] == "user"]

        if not st.session_state.conflict_resolution_mode:
            # Initial questionnaire completion
            completed_json = answer_refill(st.session_state.questions, answers, st.session_state.obs, st.session_state.bizobj)

            if check_for_conflicts(completed_json):
                # Conflicts found, start conflict resolution
                conflict_questions = question_create_conflict(completed_json)
                st.session_state.questions = ast.literal_eval(conflict_questions)
                st.session_state.current_question_index = 0
                st.session_state.conflict_resolution_mode = True
                st.write("Conflicts detected. Please answer the following questions to resolve them.")
                st.write(st.session_state.questions)

                # Display the first conflict question
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.questions[0])
                st.session_state.messages.append({"role": "assistant", "content": st.session_state.questions[0]})
                st.session_state.current_question_index += 1
            else:
                finalize_questionnaire(completed_json)
        else:
            # Conflict resolution completion
            conflict_answers = answers[-len(st.session_state.questions):]
            completed_json = answer_refill_conflict(st.session_state.questions, conflict_answers, st.session_state.obs, st.session_state.bizobj)

            if check_for_conflicts(completed_json):
                st.write("Conflicts still present. Please resolve the following conflicts.")
                conflict_questions = question_create_conflict(completed_json)
                st.session_state.questions = ast.literal_eval(conflict_questions)
                st.session_state.current_question_index = 0
                st.write(st.session_state.questions)
                show_question()  # Show the new conflict questions
            else:
                finalize_questionnaire(completed_json)


def finalize_questionnaire(completed_json):
    airtable_write(completed_json)
    exec_summ = executive_summary(completed_json)
    st.write(exec_summ)
    
    # Reset the questionnaire state
    st.session_state.questionnaire_complete = True
    st.session_state.conflict_resolution_mode = False
    chat_interaction()
    
if __name__ == "__main__":
    main()