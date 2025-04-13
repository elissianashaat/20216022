import os
import json
import requests
import tempfile
import subprocess
import pandas as pd
import pdfplumber
import chromadb
import docx2pdf
from dotenv import load_dotenv
import openai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langdetect import detect
import pickle
from typing import Optional
from sqlalchemy import create_engine, inspect


router = APIRouter()
# Add metadata to the router
route_metadata = {
    "prefix": "/api/agents/rag",
    "tags": ["RAG"]
}
load_dotenv()
DB_CONNECTION_STRING = os.getenv("DB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


class ChatRequest(BaseModel):
    file: str  # File URL
    question: str  # User's question
    instructions: Optional[str]
    # predefined_questions: list[str]

class OnPremDBRequestModel(BaseModel):
    question: str
class WFDBRequestModel(BaseModel):
    question: str
    connection_string: str

working_sheet = ''

# Logging the token consumption to the terminal (to be logged to a file later)
async def log_token_usage(response):
    """Logs token usage and updates global counter."""
    global TOTAL_TOKENS_USED
    tokens_used = response.usage.total_tokens if response.usage else 0
    TOTAL_TOKENS_USED += tokens_used
    print(f"🔹 Tokens Used: {tokens_used} | Total Tokens So Far: {TOTAL_TOKENS_USED}")

# Handle documents separately from tabular data
async def handle_pdf(file_path, question):
    """Processes PDF files using a vector store and queries OpenAI."""
    client_chroma = chromadb.PersistentClient(path="./vector_store")
    collection = client_chroma.get_or_create_collection(name="pdf_docs")

    # Get all existing document IDs and delete them
    existing_docs = collection.get()["ids"]
    if existing_docs:
        collection.delete(ids=existing_docs)  # Remove all previous documents

    with pdfplumber.open(file_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    collection.add(documents=[text], ids=[file_path])

    query_result = collection.query(query_texts=[question], n_results=5)
    retrieved_text = query_result["documents"][0][0] if query_result["documents"] else ""

    return retrieved_text



###################################################################### This part handls excel files ######################################################################

def sample_general(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        sample= {}

        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            sample_data = df.head(3).to_dict(orient="records")
            sample[sheet_name] = sample_data

        return sample
    except Exception as e:
        print(f"Error: {e}")
        return {}

async def get_working_sheet_name(question, sample_one, instructions):
    prompt = f"""
        Given this sample: {sample_one} of an excel file, return the sheet name of the sheet that has the answer to this question: {question} 
        ONLY respond with a string of the sheet name, NO further comments or eplanation.
        If No sheet seems to answer the question, default your response to the first sheet.
        Be careful as this is the most important step. Read the question and understand its semantics properly to be able to tell if the question REALLY isn't based on the sheet before you return 0.
        Sometimes the question is about max, min, average, sum, etc of a column. These values may not be explicit in the sheet, but we can still calculate them. DON'T return 0 in this case.
        Use words like "company, this company" to stand for Egyproperty, as this is the company name.
        Use the following instructions for additional context: 
        {instructions}

        If the question mentions code/number and provides alphanumeric hyphenated codes, look for column names that contain words like "id", "code", etc.

        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that determines which sheet in an excel file has the answer to a question."},
            {"role": "user", "content": prompt}
        ], 
        temperature=0
    )
    await log_token_usage(response)
    return response.choices[0].message.content.strip()

def extract_headers_and_samples(file_path, working_sheet):     
    df = pd.read_excel(file_path, sheet_name=working_sheet)
    headers = df.columns.tolist()
    sample_data = df.head(3).to_dict(orient='records')
    return headers, sample_data

async def get_relevant_columns(headers, sample, question, instructions):
    prompt = f"""
        Given this sample: {sample} of an excel sheet that has column names: {headers}, return a list of the column names that have the answer to this question: {question} 
        ONLY respond with a list of the column names, NO further comments or eplanation.
        If No column seems to answer the question, respond with an empty list. 
        If the question mentions code/number and provides a alphanumeric hyphenated codes, look for column names that contain words like "id", "code", etc.
        Use the following instructions for additional context:
        {instructions}
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that determines which columns in an excel sheet have the answer to a question."},
            {"role": "user", "content": prompt}
        ], 
        temperature=0
    )
    await log_token_usage(response)
    return response.choices[0].message.content.strip()
    
def run_script(file_path, script_path):
    global working_sheet
    try:
        sheet_dataframe = pd.read_excel(file_path, sheet_name=working_sheet)
        # Save DataFrame as a pickle file
        with open("data.pkl", "wb") as f:
            pickle.dump(sheet_dataframe, f)
        result = subprocess.run(["python", script_path, "data.pkl"], capture_output=True, text=True)
        extracted_row = result.stdout.strip() if result.returncode == 0 else json.dumps({"error": "Script execution failed"})
        return extracted_row

    except Exception as e:
        print(f'error: {e}')
        return {}

def extract_categorical_values(file_path, working_sheet, uniqueness_threshold=0.95, max_unique_ratio=0.02, max_unique_values=20):
    try:
        extracted_data = {}
        df = pd.read_excel(file_path, sheet_name=working_sheet)
        total_rows = len(df)
        unique_counts = df.nunique()
        high_uniqueness_columns = unique_counts[unique_counts >= (len(df) * uniqueness_threshold)].index.tolist()
        
        filtered_values = {
            col: df[col].dropna().unique().tolist()
            for col in df.select_dtypes(include=['object', 'category']).columns
            if col not in high_uniqueness_columns and unique_counts[col] <= max(total_rows * max_unique_ratio, max_unique_values)
        }
        
        if filtered_values:
            extracted_data = filtered_values
                
        return extracted_data
    except Exception as e:
        print(f"Error: {e}")
        return {}

async def get_unique_categories(extracted_values, query, instructions):
  prompt = f"""
            You are a data extraction assistant. Your goal is to cooperate with other agents to narrow fown the search in the following scheme:
            There are several steps to add layers of context to find data easily in an excel sheet. Your task is to add to that context be making a simple match.
            The columns with highly repeating values are extracted and reduced down to the unique values that occur in each column as: {extracted_values}. 
            If there is ANY of these unique values match the intent of the question: {query}, respond with a list of these cell values as a string. 
            Respond ONLY with the list, no further comments or explanation.
            If no cell value matches any keywords, return an empty list.
            Use the following instructions for additional context clarit: 
            {instructions}
            """

  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": "Return ONLY the list of unique values that best match the keywords in the question."},
          {"role": "user", "content": prompt}
      ], 
      temperature=0
  )
  await log_token_usage(response)
  return response.choices[0].message.content.strip()

async def generate_extraction_script(question, sample_data, headers, relevant_columns, categorical_values, relevant_cell_values, instructions):
    prompt = f"""You are an AI agent that generates a Python script to process an excel sheet and extract rows relevant to the answer of the question: {question}. 
    Take the following layers of context into account to know what data are inside the sheet and how they look. 
    These layers are helping steps, some of them might actually add to the context depending on the questio, so if any of the variables is empty, ignore it in the fliteration/extratction process.
    
    - Sample of the data for reference (THIS IS NOT THE FULL DATA): {sample_data}
    - All column names of the entire sheet: {headers}
    - Names of the columns that are relevant to the answer: {relevant_columns}
    - The columns that have highly repeating values reduced down to only the unique values in each column (usually these are key/important columns): {categorical_values}
    - Cell values in the highly repeating columns that match keywords in the question : {relevant_cell_values}

    Instructions:
    - Determine whether to extract values directly or apply mathematical operations.
    - Use Pandas to filter data based on the context above.
    - Ensure proper data type conversion (If you need to compare cell values to a string, use astype(str) for conversion instead of .str (string accessor)).
    - The response MUST ONLY be the script with NO extra text, markdown, or comments/explanation.
    - There will already be a dataframe containing the sheet your script will work on. This dataframe will be passed to the subprocess as a pickle file, 
        so make sure to import pickle and read the first argument into a df variable and then execute the filteration/extraction logic.
    - if the answer is not an aggregation of values, make sure to ALWAYS return the entire row that contains the answer.
    - covert all cell text to lower case before comparing or filtering/matching.
    Your script MUST use an existing dataframe in the local code, so DON'T create a variable that stores any data for the script to run on.
    Use the following instructions for additional context claarity: 
    {instructions}
    Always convert the extracted row to a string before returning it.

    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate a Python script with NO comments, markdown, or extra text. Only return valid, executable Python code."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    await log_token_usage(response)
    
    return response.choices[0].message.content.strip()

async def generate_friendly_response(question, extracted_text, instructions=""):
    """Asks OpenAI to rephrase the extracted text naturally."""
    detected_lang = detect(question)
    response_lang = "in Egyptian Arabic" if detected_lang == "ar" else "in English"

    prompt = f"""
    You are a helpful AI assistant that rephrases answers into a friendly natural language answer. A user asked:
    "{question}"

    The extracted row/text for the answer is:
    {extracted_text}

    Rephrase this into a natural, conversational response {response_lang}. 
    If the extracted answer row/text is empty, just say you don't know the answer yet and offer help with another question. DON'T answer the question on your own, ALWAYS use extracted row/text ONLY for the answer.
    If the question is in Egyptian Araic and you couldn't help with the answer, remind the uset to type company and project names in English for better matching.
    If the question is in Arabic, always answer in Egyptian Arabic, Otherwise, use English.
    Further important instructions:
    {instructions}
    """
    # Here are further instructtions: {instructions}

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )
    await log_token_usage(response)

    return response.choices[0].message.content

############################################################################ end of the part ################################################################################



###################################################################### This part handles databases ##########################################################################

def get_db_metadata(connection_string: str):
    """Extract database metadata including tables and columns."""
    try:
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        
        metadata = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            metadata[table_name] = {col['name']: col['type'].__str__() for col in columns}
        print(f"meta: {metadata}")
        return metadata
    except Exception as e: 
        print(f"failed to connect: {e}")

def generate_sql_query(metadata: dict, question: str):
    """Generate an SQL query using OpenAI based on metadata and user question."""
    prompt = f"""
    Given the following database schema:
    {json.dumps(metadata, indent=2)}
    
    Write a PSQL query to answer the question: "{question}"
    Rspond with ONLY the psql query, no further comments or explanation.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that generates sql queries"},
            {"role": "user", "content": prompt}
        ], 
        temperature=0
    )
    print(f"\n\n\n script: {response.choices[0].message.content.strip('```sql').strip('```')}")
    
    return response.choices[0].message.content.strip("```sql").strip("```")

def execute_query(connection_string: str, query: str):
    """Execute the SQL query using a subprocess and return the output."""
    engine = create_engine(connection_string)
    db_type = engine.dialect.name
    
    if db_type == "postgresql":
        cmd = ["psql", connection_string, "-c", query]
    elif db_type == "mysql":
        cmd = ["mysql", "-e", query]
    elif db_type == "sqlite":
        db_path = connection_string.replace("sqlite:///", "")
        cmd = ["sqlite3", db_path, query]
    else:
        raise ValueError("Unsupported database type")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(result.stderr)
    
    return result.stdout

############################################################################ end of the part ################################################################################



# endpoint to chat with files
@router.post("/chat")
async def extract_data(request: ChatRequest):
    """
    Handles file download, processes tabulated data, processes PDFs using vector stores,
    and retrieves answers from OpenAI in the appropriate language.
    """
    global TOTAL_TOKENS_USED
    TOTAL_TOKENS_USED = 0

    file_url, question, instructions = request.file, request.question, request.instructions
    file_ext = file_url.split(".")[-1].lower()

    if file_ext not in ["xls", "xlsx", "csv", "pdf", "doc", "docx"]:
        raise HTTPException(status_code=400, detail="Only CSV, Excel, PDF, DOC, and DOCX files are supported.")

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, f"uploaded_file.{file_ext}")

    try:
        with requests.get(file_url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as buffer:
                for chunk in response.iter_content(chunk_size=8192):
                    buffer.write(chunk)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download the file: {str(e)}")

    if file_ext == "csv":
        excel_path = file_path.replace(".csv", ".xlsx")
        df = pd.read_csv(file_path)
        df.to_excel(excel_path, index=False, engine='openpyxl')
        file_path = excel_path
        file_ext = "xlsx"

    
    if file_ext in ["doc", "docx"]:
        pdf_path = file_path.replace(file_ext, "pdf")
        docx2pdf.convert(file_path, pdf_path)
        file_path = pdf_path
        file_ext = "pdf"

    if file_ext == "pdf":
        friendly_response = await handle_pdf(file_path, question)
    else:
        sample_one = sample_general(file_path)
        global working_sheet 
        working_sheet = await get_working_sheet_name(question, sample_one, instructions)
        headers, sample_data = extract_headers_and_samples(file_path, working_sheet)
        relevant_columns = await get_relevant_columns(headers, sample_data, question, instructions)
        categorical_values =  extract_categorical_values(file_path, working_sheet)
        relevant_cell_values = await get_unique_categories(categorical_values, question, instructions)
        script = await generate_extraction_script(question, sample_data, headers, relevant_columns, categorical_values, relevant_cell_values, instructions)
        clean_script = script.strip("```python").strip("```")
        script_path = os.path.join(temp_dir, "extract_script.py")
        with open(script_path, "w", encoding='utf-8') as script_file:
            script_file.write(clean_script)
        extracted_row = run_script(file_path, script_path)
        friendly_response = await generate_friendly_response(question, extracted_row, instructions)

    print(f"🚀 **Total Tokens Used for This Request: {TOTAL_TOKENS_USED}**")
    return {"friendly_response": friendly_response, "total_tokens_used": TOTAL_TOKENS_USED}

# endpoint to chat with database on prem
@router.post("/on_prem_db")
def query_db(request: OnPremDBRequestModel):
    try:
        metadata = get_db_metadata(DB_CONNECTION_STRING)
        sql_query = generate_sql_query(metadata, request.question)
        result = execute_query(DB_CONNECTION_STRING, sql_query)
        return {"query": sql_query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# endpoint to chat with database in a workflow
@router.post("/wf_db")
def wf_db(request: WFDBRequestModel):
    try:
        metadata = get_db_metadata(request.connection_string)
        sql_query = generate_sql_query(metadata, request.question)
        result = execute_query(request.connection_string, sql_query)
        return {"query": sql_query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))