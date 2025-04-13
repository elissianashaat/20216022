from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid, time
from dotenv import load_dotenv
import os 
from sentence_transformers import SentenceTransformer
import json
import openai

# Load environment variables
load_dotenv()

router = APIRouter()
route_metadata = {
    "prefix": "/api/agents/chat-sequencer",
    "tags": ["chat sequencer", "chatbot"]
}

# Set env variables and initialize DB and AI clients
qdrant_key = os.getenv("QDRANT_API_KEY")
cluster_url = os.getenv("QDRANT_CLUSTER_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
ai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
client = QdrantClient(
    url=cluster_url, 
    api_key=qdrant_key,
)
print("✅ Connected to Qdrant Cloud with REST client")

# Define request models
class RephraseRequest(BaseModel):
    query: str
    user_id: str
    sys_msg: str
class RAGRequest(BaseModel):
    objective: str
    query: str
    rephrased_query: str
    rag_output: str
    history: list[dict]
    user_id: str
    instructions: str
    profile_template: str

class NonRagRequest(BaseModel):
    objective: str
    query: str
    history: list[dict]
    user_id: str
    instructions: str
    profile_template: str

class RerouteRequest(BaseModel):
    query: str
    history: List[dict]

class SummaryRequest(BaseModel):
    history: list[dict]
    user_id: str

# Define response models
class RephraseResponse(BaseModel):
    rephrased_query: str

class RAGResponse(BaseModel):
    response: str
    isObjMet: bool

class NonRagResponse(BaseModel):
    response: str
    isObjMet: bool

class RerouteResponse(BaseModel):
    is_rag: bool

class SummaryResponse(BaseModel):
    summary: str




########################################################################### Database and memory management ########################################################################

collection_name = "chat_memory"
# Check and create only if not exists
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# Specifying the model used for vector embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store new Data 
def store_message(user_id, speaker, text):
    vector = model.encode(text).tolist()
    message_id = str(uuid.uuid4())
    timestamp = int(time.time())

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=message_id,
                vector=vector,
                payload={
                    "user_id": user_id,
                    "speaker": speaker,
                    "text": text,
                    "timestamp": timestamp,
                    "type": "message"
                }
            )
        ]
    )

def update_user_summary(user_id: str, query: str, sys_msg: str, profile_template: str = '') -> List[str]:


    # Step 1: Retrieve the current summary for this user (if any)
    summary_filter = Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="type", match=MatchValue(value="summary"))
        ]
    )

    summaries = client.scroll(
        collection_name=collection_name,
        scroll_filter=summary_filter,
        limit=1
    )[0]
    print(f"old summaries: {summaries}")

    previous_summary = summaries[0].payload["text"] if summaries else "{}"
    point_id = summaries[0].id if summaries else str(uuid.uuid4())

    # Step 2: Generate prompt for updating summary
    prompt = f"""
    Here is the profile template::
    {profile_template}

    Here is the current JSON user profile:
    {previous_summary}

    The last system message was: 
    "{sys_msg}"
    The user responded with:
    "{query}"

    Update the profile. Keep it in proper JSON format and include any new details, replacing older ones if needed.
    Be careful when replacing old data: don't infer updates to the profile from the new messages unless it's clear that these are updates.
    Example 1: the user asks for a car priced around 200,000 and their profile has a budget field of 500,000. In that case you shouldn't update the budget to 200,000 because the new message doesn't explicitly state this as a budget.
    Example 2: the user provides their email as john@example.me and their profile has a name field of Alice. In that case you shouldn't update the name to John.

    Only include information relevant to their identity, preferences, and buying intent.
    Respond with only the updated JSON, no further comments or explanation.
    """ # We should inject a user profile template into the prompt

    # Step 3: Call OpenAI to update the summary
    response = ai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that creates and/or updates a user profile based on their conversation with the system."},
            {"role": "user", "content": prompt}
        ]
    )

    updated_summary = response.choices[0].message.content.strip("```json").strip("```")

    # Step 4: Embed and store updated summary
    vector = model.encode(updated_summary).tolist()
    timestamp = int(time.time())

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "user_id": user_id,
                    "type": "summary",
                    "text": updated_summary,
                    "timestamp": timestamp,
                    "speaker": "system"
                }
            )
        ]
    )

    print(f"✅ Updated summary for user {user_id}")
    return [summaries, updated_summary]

# Retrieve old data
def retrieve_conversation_pairs(user_id, query, top_k=3):
    query_vector = model.encode(query).tolist()
    memory_filter = Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="type", match=MatchValue(value="message"))
        ]
    )


    # Step 1: Qdrant search with vectors
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=memory_filter,
        limit=top_k,
        with_vectors=False  # Not needed since we won't re-rank
    )
 
    print("🔍 Qdrant top-k search results:", len(results))

    # Step 2: Load full user memory
    all_points = client.scroll(
        collection_name=collection_name,
        scroll_filter=memory_filter,
        limit=500  # Adjust as needed
    )[0]

    # Step 3: Sort full memory chronologically
    all_sorted = sorted(all_points, key=lambda x: x.payload["timestamp"])

    print(f"RAG results: {all_sorted}")
    # Step 4: For each result, find the message and its context
    conversation_blocks = []
    for r in results:
        ts = r.payload["timestamp"]
        txt = r.payload["text"]

        idx = next((i for i, pt in enumerate(all_sorted) if pt.payload["timestamp"] == ts and pt.payload["text"] == txt), None)
        if idx is None:
            continue

        block = []

        # Previous message
        if idx - 1 >= 0:
            prev = all_sorted[idx - 1].payload
            block.append(f"{prev['speaker'].capitalize()}: {prev['text']}")

        # Current message
        curr = all_sorted[idx].payload
        block.append(f"{curr['speaker'].capitalize()}: {curr['text']}")

        # Next message
        if idx + 1 < len(all_sorted):
            nxt = all_sorted[idx + 1].payload
            block.append(f"{nxt['speaker'].capitalize()}: {nxt['text']}")

        conversation_blocks.append("\n".join(block))

    return "\n\n".join(conversation_blocks)

############################################################################### end of memory part ############################################################################

class AIAgent:
    def __init__(self, api_key: str, threshold: float = 0.75):

        self.llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=api_key)
        self.threshold = threshold
        
        # LLM templates
        self.task_template = PromptTemplate(
            input_variables=["objective"],
            template="""
            divide this objective: "{objective}" into a numbered list that cover up all parts of the objective. For example:
            the objective "get the user's email, phone number, and budget" should be divided into:
            1. get the user's email
            2. get the user's phone number
            3. get the user's budget
            If the objective includes only one task, write it as it is but numbered 1.
            DON'T include any additional information or instructions in the response, simply stick to what is explicitly stated in the objective.
            """
        )
        self.task_status_template = PromptTemplate(
            input_variables=["tasks", "history", "query", "sys_msg"],
            template="""
            You are an assistant that helps a chatbot assistant with tracking the status of the objective of the conversation. 
            The objective is broken down into a list of tasks as follows: {tasks}
            We retrieved the relevant parts of the chat history from the vectore store and they are: {history}.
            In addition to the history, here is the last systme message: {sys_msg}
            to which the user responded with: {query}
            Respond briefly with what parts of the tasks are done and what remain.
            """
        )
        self.non_rag_template = PromptTemplate(
        input_variables=["task_status", "query", "history", "user_info"],
        template="""
            You are a chatbot assistant that provide customer support to users and answer their questions in a brief and friendly way. The conversations are not loose, we made sure to provide an objective for each conversation. This objective is broken down into a list of tasks.
            The status of the tasks: {task_status}
            User msg: {query}
            user profile: {user_info}
            Read the status of tasks and respond to the user to get the remaining tasks done. 
            Your response should:
            1. Match the language of the query and user's sentiment if any.
            2. Be conversational and friendly, but concise and to the point.
            3. Maintain cultural appropriateness
            4. Use the history to evaluate which tasks are achieved and NEVER confirm or return to the tasks that ar achieved already.
            If the query is a feedback/issue report, let the user know that a human will see their feedback and ask them for contact information if needed.
            Pay attention to the status of tasks and NEVER EVER repeat a task that has been done or confirm the information provided to you. ALWAYS assume that the data the user provides are confirmed and valid. 
            Once all tasks are done, Never offer additional help, but just thank the user, as they will be directed to another conversation flow. 
            """
        )
        self.query_routing_template = PromptTemplate(
        input_variables=["query", "history"],
        template="""
            Read the conversation history between the user ans the system: {history}
            The last user message is: "{query}" 
            Determine if the the user's query is to be answered by the RAG system or it's not requesting information (it's a sentiment or feedback for example).
            repond with ONLY "True" if the query is requesting information that can be answered by the RAG system, otherwise respond with "False".
    
            """
        )
        self.convo_summary_template = PromptTemplate(
        input_variables=["history", "user_info"],
        template="""
            Summarize the following conversation history in a few sentences. Focus on extracting info user provided about themselves (e.g. name, contact info, preferences, budget, interest, etc)
            convo history: {history}
            user info: {user_info}
            """
        )
        self.judge_objective_template = PromptTemplate(
        input_variables=["tasks_status", "obj"],
        template="""
            You are an assistant that judges whether a conversation's objective is achieved or not. The objective has been broken down into a list of tasts. Another assistant evaluated and summarized which tasks have been done and which remain. Use the task status: "{task_status}" to judge if the if the objective: "{obj}" is met.
            respond ONLY with either "True" or "False"
            """
        )
        self.rephrase_query_template = PromptTemplate(
        input_variables=["history", "query", "summary"],
        template="""
        You are an assistant that disambiguates user's messages before their sent to the RAG so that the msg is clear and answerable, not obsecure or references parts that won't be sent to RAG. The only thing that will be sent to RAG is your response so make sure everything is explicitly stated.
        Read the retrieved conversation history between the user and the system: {history}
        and the user's updated profile: {summary}
        The user lastly said: "{query}" 
        Rephrase the user's message so that it's clear and answerable without looking at the history or profile. Include everything in the updated profile into the rephrased question and make it very brief and concise. 
        ALWAYS match the user's Language, either English or Egyptian Arbic.
        If the query doesn't reference any part of the history and is clear, return the query as it is.
        Respond only with the query. 
        """
        )
        self.clean_rag_template = PromptTemplate(
            input_variables=["task_status", "query", "rag_output"],
            template="""
            You are an AI assistant that cleans the RAG system output and turns it into a chatbot response to the user. We also want the response to force the user towards achieving the remaining task:
            
            Status of the tasks to follow: {task_status}
            User Query: {query}
            RAG Output: {rag_output}

            Modify the RAG output by replacing any closing sentences with "do you have any other questions?" (or something similar) to force the user to achieve the tasks with the system.
            
            Respond ONLY with the modified RAG output.
            """
        )
        self.request_info_template = PromptTemplate(
        input_variables=["history", "query", "summary"],
        template="""
            Read the recent conversation between the user and the system: {history}
            and the latest user question: {query}
            The user profile: {summary}
            Your task is to check if the user assumes we have any data, but we don't have it in the history or the user profile. If we don't have the data in question, prompt the user to provide ONLY that data and nothing more. 
            Your response should be JSON of the form "prompt": "<the question we need to ask the user to get missing info>", "isMissing": true/false
            If we have all the data, return "prompt": "", "isMissing": false.
            """
        )
        # LLM chains
        self.task_chain = self.task_template | self.llm
        self.task_status_chain = self.task_status_template | self.llm
        self.non_rag_chain = self.non_rag_template | self.llm
        self.query_routing_chain = self.query_routing_template | self.llm
        self.convo_summary_chain = self.convo_summary_template | self.llm
        self.judge_objective_chain = self.judge_objective_template | self.llm
        self.rephrase_query_chain = self.rephrase_query_template | self.llm
        self.request_info_chain = self.request_info_template | self.llm
        self.clean_rag_chain = self.clean_rag_template | self.llm

    def objective_to_tasks(self, objective: str) -> List[str]:
        tasks_raw = self.task_chain.invoke({"objective": objective})
        tasks = [task.strip() for task in tasks_raw.content.split('\n') if task.strip()]
        return tasks
    
    def get_task_status(self, tasks: str, history: str,sys_msg: str, query: str) -> str:
        res = self.task_status_chain.invoke({
            "tasks": tasks,
            "history": history,
            "sys_msg": sys_msg,
            "query": query
        })

        return res.content.strip()
    
    def respond_rag(self, task_status: str, query: str, rag_output: str) -> Dict:
        
        analysis = self.clean_rag_chain.invoke({
            "task_status": task_status,
            "query": query,
            "rag_output": rag_output
        })
        res = analysis.content.strip("```json").strip("```")
        
        return res
    
    def route_query(self, query, history) -> bool:
        # Determine if the query should be answered by the RAG system
        response = self.query_routing_chain.invoke({"query": query, "history": history})
        return response.content.strip() == "True"

    def respond_non_rag(self, status: str, query: str, user_info: dict) -> str:
        # Generate tasks from objective
        
        # Get LLM analysis
        analysis = self.non_rag_chain.invoke({
            "task_status": status,
            "query": query,
            "user_info": user_info
        })
        res = analysis.content.strip("```json").strip("```")
        
        return res
    
    def convo_summary(self, history: list[dict], user_info: dict) -> str:
            # Get LLM analysis
            analysis = self.convo_summary_chain.invoke({
                "history": history,
                "user_info": user_info
            })
            return analysis.content

    def judge_objective(self, task_status: str, obj: str) -> bool:
        judgement = self.judge_objective_chain.invoke({
            "task_status": task_status,
            "obj": obj
        })
        return judgement.content.strip().lower() == "true"

    def rephrase_query(self, history: str, query: str, summary) -> str:
        res = self.rephrase_query_chain.invoke({
            "history": history,
            "query": query,
            "summary": summary
        })

        return res.content.strip()

    def request_info(self, history: List[dict], query: str, user_info: dict) -> Dict:
        res = self.request_info_chain.invoke({
            "history": history,
            "query": query,
            "summary": user_info
        })
        return json.loads(res.content.strip('```json').strip('```').strip())
    
    def log_info(self, history: List[dict], current_state: dict) -> Dict:
        res = self.user_info_chain.invoke({
            "history": history,
            "user_info": current_state
        })
        return json.loads(res.content.strip('```json').strip('```').strip())

# Initialize the AIAgent
agent = AIAgent(OPENAI_API_KEY)

@router.post("/rephrase", response_model=RephraseResponse)
def rephrase_queries(request: RephraseRequest):
    query, user_id, sys_msg = request.query, request.user_id, request.sys_msg
    _, summary = update_user_summary(user_id, query, sys_msg)
    semantic_history = retrieve_conversation_pairs(user_id, query, 8)
    rephrased_query = agent.rephrase_query(semantic_history, query, summary)
    return {"rephrased_query": rephrased_query}

@router.post("/reroute", response_model=RerouteResponse)
async def reroute_query(request: RerouteRequest):
    try:
        is_rag = agent.route_query(request.query, request.history)
        return {"is_rag": is_rag}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag", response_model=RAGResponse)
async def rag(request: RAGRequest):
    try:
        ## we want to modify RAG output and judge objective
        obj = request.objective
        query = request.query
        user_id = request.user_id
        temp = request.profile_template
        sys_msg = request.history[-1]["message"] if len(request.history) > 0 else ""
        rag_output = request.rag_output

        result = {}
        tasks = agent.objective_to_tasks(obj)
        semantic_history = retrieve_conversation_pairs(user_id, query, 8)
        task_status = agent.get_task_status(tasks, semantic_history, sys_msg, query)
        _, summary = update_user_summary(user_id, query, sys_msg, temp)
        
        request_info = agent.request_info(semantic_history, query, summary)
        isMissingInfo: bool = request_info["isMissing"]
        print(f"is missing info: {isMissingInfo}")
        prompt = request_info["prompt"]
        if not isMissingInfo: 
            result["response"] = agent.respond_rag(task_status, query, rag_output)
            result["isObjMet"] = agent.judge_objective(task_status, obj)
            store_message(user_id, "user", query)
            store_message(user_id, "bot", result["response"])
            return result
        result["response"] = prompt
        result["isObjMet"] = False
        store_message(user_id, "user", query)
        store_message(user_id, "bot", result["response"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/convo_summary", response_model=SummaryResponse)
async def convo_summary(request: SummaryRequest):
    try:
        summary_filter = Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value=request.user_id)),
            FieldCondition(key="type", match=MatchValue(value="summary"))
        ]
    )

        summaries = client.scroll(
            collection_name="chat_memory",
            scroll_filter=summary_filter,
            limit=1
        )[0]

        summary = summaries[0].payload["text"] if summaries else "{}"
        result = {}
        res = agent.convo_summary(
            history=request.history,
            user_info=summary
        )
        result["summary"] = res + json.dumps(summary)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

@router.post("/non-rag", response_model=NonRagResponse)
async def non_rag(request: NonRagRequest):
    try:
        obj = request.objective
        query = request.query
        user_id = request.user_id
        temp = request.profile_template
        sys_msg = request.history[-1]["message"] if len(request.history) > 0 else ""
        tasks = agent.objective_to_tasks(obj)
        semantic_history = retrieve_conversation_pairs(user_id, query, 3)
        task_status = agent.get_task_status(tasks, semantic_history, sys_msg, query)


        result = {}
        [old_summary, summary] = update_user_summary(user_id, query, sys_msg, temp)
        res = agent.respond_non_rag(
            status=task_status,
            query=query,
            user_info=summary
        )

        result["response"] = res
        isObjMet = agent.judge_objective(task_status, obj)
        result["isObjMet"] = isObjMet
        
        store_message(user_id, "user", query)
        store_message(user_id, "bot", res)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


