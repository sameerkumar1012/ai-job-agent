import requests
import os
import json
import asyncio

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

from pypdf import PdfReader

from telegram import Bot

from langchain_groq import ChatGroq

import google.generativeai as genai

from apscheduler.schedulers.blocking import (
    BlockingScheduler
)

# ===================================================
# LOAD ENV VARIABLES
# ===================================================

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

# ===================================================
# CONFIGURE GEMINI
# ===================================================

genai.configure(
    api_key=GEMINI_API_KEY
)

gemini_model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)

# ===================================================
# CONFIGURE GROQ
# ===================================================

groq_llm = ChatGroq(
    api_key = GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

# ===================================================
# TELEGRAM BOT
# ===================================================

bot = Bot(
    token=TELEGRAM_BOT_TOKEN
)

# ===================================================
# LOAD EMBEDDING MODEL
# ===================================================

print(
    "\nLoading Embedding Model...\n",
    flush=True
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ===================================================
# TELEGRAM FUNCTION
# ===================================================

async def send_telegram_message(message):

    try:

        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message
        )

        print(
            "\nTelegram Alert Sent!\n",
            flush=True
        )

    except Exception as e:

        print(
            "\nTelegram Error:\n",
            flush=True
        )

        print(e)

# ===================================================
# MAIN JOB SEARCH FUNCTION
# ===================================================

def run_job_search():

    print(
        "\n===================================",
        flush=True
    )

    print(
        "\nStarting AI Job Search...\n",
        flush=True
    )

    # ===============================================
    # LOAD USER PROFILE
    # ===============================================

    with open(
        "user_profile.json",
        "r"
    ) as file:

        profile = json.load(file)

    # ===============================================
    # LOAD SEEN JOBS
    # ===============================================

    if os.path.exists(
        "seen_jobs.json"
    ):

        with open(
            "seen_jobs.json",
            "r"
        ) as file:

            seen_jobs = set(
                json.load(file)
            )

    else:

        seen_jobs = set()

    # ===============================================
    # LOAD RESUME PDFS
    # ===============================================

    resume_text = ""

    data_folder = "data"

    print(
        "\nLoading Resume PDFs...\n",
        flush=True
    )

    for file in os.listdir(data_folder):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(
                data_folder,
                file
            )

            print(
                f"Loading Resume: {file}",
                flush=True
            )

            reader = PdfReader(pdf_path)

            for page in reader.pages:

                text = page.extract_text()

                if text:

                    resume_text += text

    # ===============================================
    # CREATE RESUME EMBEDDING
    # ===============================================

    print(
        "\nCreating Resume Embedding...\n",
        flush=True
    )

    resume_embedding = model.encode(
        [resume_text]
    )

    # ===============================================
    # DYNAMIC SEMANTIC SEARCH QUERIES
    # ===============================================

    location = profile["location"]

    search_queries = [

        f"Data Analytics fresher jobs in {location}",

        f"Business Analyst fresher jobs in {location}",

        f"Python SQL fresher jobs in {location}",

        f"Tableau Power BI fresher jobs in {location}",

        f"AWS Data fresher jobs in {location}",

        f"ETL fresher jobs in {location}",

        f"Analytics Associate fresher jobs in {location}",

        f"Insights Analyst fresher jobs in {location}",

        f"Operations Analyst fresher jobs in {location}",

        f"Reporting Analyst fresher jobs in {location}"
    ]

    # ===============================================
    # JSEARCH API CONFIG
    # ===============================================

    url = "https://jsearch.p.rapidapi.com/search"

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }

    # ===============================================
    # LOOP THROUGH QUERIES
    # ===============================================

    for semantic_search_query in search_queries:

        print(
            f"\nSearching Query:\n"
            f"{semantic_search_query}\n",
            flush=True
        )

        querystring = {

            "query": semantic_search_query,

            "page": "1",

            "num_pages": "1"
        }

        response = requests.get(
            url,
            headers=headers,
            params=querystring
        )

        data = response.json()

        print(
            f"\nJobs Returned: "
            f"{len(data.get('data', []))}\n",
            flush=True
        )

        if "data" not in data:

            continue

        # ===========================================
        # PROCESS JOBS
        # ===========================================

        for job in data["data"]:

            title = job.get(
                "job_title",
                "N/A"
            )

            company = job.get(
                "employer_name",
                "N/A"
            )

            location = job.get(
                "job_city",
                "N/A"
            )

            description = job.get(
                "job_description",
                ""
            )

            apply_link = job.get(
                "job_apply_link",
                "N/A"
            )

            posted_date = job.get(
                "job_posted_at_datetime_utc",
                "N/A"
            )

            job_id = job.get(
                "job_id",
                company + title
            )

            # =======================================
            # DEDUPLICATION
            # =======================================

            if job_id in seen_jobs:

                continue

            # =======================================
            # EXPERIENCE FILTER
            # =======================================

            description_lower = description.lower()

            blocked_keywords = [

                "2+ years",
                "3+ years",
                "4+ years",
                "5+ years",

                "minimum 2 years",
                "minimum 3 years",

                "2 years experience",
                "3 years experience",

                "senior",
                "lead",
                "manager",
                "architect",

                "experienced professional",
                "prior experience required",

                "industry experience",
                "relevant experience required"
            ]

            if profile["experience"] == "fresher":

                if any(
                    keyword in description_lower
                    for keyword in blocked_keywords
                ):

                    continue

            # =======================================
            # SEMANTIC MATCHING
            # =======================================

            print(
                f"Checking Match: {title}",
                flush=True
            )

            job_embedding = model.encode(
                [description]
            )

            similarity = cosine_similarity(
                resume_embedding,
                job_embedding
            )[0][0]

            match_score = round(
                similarity * 100,
                2
            )

            print(
                f"Match Score: "
                f"{match_score}%",
                flush=True
            )

            # =======================================
            # FILTER LOW MATCHES
            # =======================================

            if match_score < 50:

                continue

            # =======================================
            # GEMINI VERIFICATION
            # =======================================

            print(
                "\nRunning Gemini Verification...\n",
                flush=True
            )

            verification_prompt = f"""
Analyze this job description carefully.

Determine how suitable this role is for:
- freshers
- entry-level candidates
- 0 years experience

Return ONLY a number from 1 to 10.

Where:
1 = not suitable for freshers
10 = perfect fresher role

Job Description:
{description[:3000]}
"""

            try:

                verification_response = (
                    gemini_model.generate_content(
                        verification_prompt
                    ).text.strip()
                )

                try:

                    fresher_score = int(
                        verification_response[0]
                    )

                except:

                    fresher_score = 7

            except Exception as e:

                print(
                    "\nGemini Error:\n",
                    flush=True
                )

                print(e)

                fresher_score = 7

            print(
                f"Fresher Score: "
                f"{fresher_score}/10",
                flush=True
            )

            # =======================================
            # STRICT FRESHER FILTER
            # =======================================

            if fresher_score < 5:

                continue

            # =======================================
            # GROQ SUMMARY
            # =======================================

            print(
                "\nGenerating Groq Summary...\n",
                flush=True
            )

            summary_prompt = f"""
You are an AI career assistant.

Analyze the candidate resume and
job description.

Resume:
{resume_text[:2000]}

Job Description:
{description[:2000]}

Give:
1. Matching skills
2. Missing skills
3. Why suitable for fresher

Keep response concise.
"""

            try:

                summary_response = (
                    groq_llm.invoke(
                        summary_prompt
                    ).content
                )

            except Exception as e:

                print(
                    "\nGroq Error:\n",
                    flush=True
                )

                print(e)

                summary_response = (
                    "Summary unavailable."
                )

            # =======================================
            # FINAL OUTPUT
            # =======================================

            print(
                "\n================================",
                flush=True
            )

            print(
                f"Company: {company}",
                flush=True
            )

            print(
                f"Role: {title}",
                flush=True
            )

            print(
                f"Location: {location}",
                flush=True
            )

            print(
                f"Match Score: "
                f"{match_score}%",
                flush=True
            )

            print(
                f"Fresher Score: "
                f"{fresher_score}/10",
                flush=True
            )

            print(
                f"Posted Date: "
                f"{posted_date}",
                flush=True
            )

            print(
                "\nAI Summary:\n",
                flush=True
            )

            print(
                summary_response,
                flush=True
            )

            print(
                f"\nApply Link:\n"
                f"{apply_link}",
                flush=True
            )

            print(
                "================================\n",
                flush=True
            )

            # =======================================
            # SAVE JOB
            # =======================================

            seen_jobs.add(job_id)

            with open(
                "seen_jobs.json",
                "w"
            ) as file:

                json.dump(
                    list(seen_jobs),
                    file
                )

            # =======================================
            # TELEGRAM MESSAGE
            # =======================================

            telegram_message = f"""
🚀 New AI Fresher Job Match

🏢 Company: {company}

💼 Role: {title}

📍 Location: {location}

🎯 Match Score: {match_score}%

🧑 Fresher Score:
{fresher_score}/10

📅 Posted:
{posted_date}

🔗 Apply:
{apply_link}
"""

            print(
                "\nSending Telegram Alert...\n",
                flush=True
            )

            loop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)

            loop.run_until_complete(
                send_telegram_message(
                    telegram_message
                )
            )

            loop.close()

    print(
        "\nJob Search Completed!\n",
        flush=True
    )

# ===================================================
# SCHEDULER
# ===================================================

scheduler = BlockingScheduler()

scheduler.add_job(
    run_job_search,
    "interval",
    hours=8
)

print(
    "\nAI Job Agent Running...\n",
    flush=True
)

# RUN IMMEDIATELY

run_job_search()

# START SCHEDULER

scheduler.start()
