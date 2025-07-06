# from flask import Flask, request, jsonify
# import google.generativeai as genai
# from google.generativeai import GenerativeModel
# from serpapi import GoogleSearch
# import requests
# from bs4 import BeautifulSoup
# from langchain.document_loaders import WebBaseLoader
# from urllib.parse import urlparse
# import mimetypes
# import PyPDF2
# import re
# import json
# import os
# import time

# from dotenv import load_dotenv
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

# app = Flask(__name__)

# # --- API KEYS ---
# load_dotenv()
# serp_api_key = os.environ.get("SERP_API_KEY")
# gemini_key = os.environ.get("GEMINI_API_KEY")

# genai.configure(api_key=gemini_key)
# model = GenerativeModel('gemini-2.0-flash-lite')

# # --- Selenium Setup ---
# options = Options()
# options.add_argument("--headless")
# options.add_argument("--disable-gpu")
# options.add_argument("--no-sandbox")
# driver = webdriver.Chrome(options=options)

# # --- Utility Functions ---
# def is_webpage(url):
#     try:
#         response = requests.head(url, allow_redirects=True)
#         return 'text/html' in response.headers.get('Content-Type', '').lower()
#     except requests.exceptions.RequestException:
#         return False

# def clean_text(text, max_length=16000):
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text[:max_length]

# def scrape_and_summarize(url, max_length=16000):
#     if not is_webpage(url):
#         return ""
#     loader = WebBaseLoader(web_paths=[url])
#     docs = loader.load()
#     return clean_text(docs[0].page_content, max_length) if docs else ""

# def gemini_model_summary(title, content, topic):
#     if not content.strip():
#         return {"summary": "", "content_index": 0}
#     prompt = f"""
# You are an AI assistant analyzing article content to assess its relevance to a given topic and extract a high-quality summary.

# Your tasks:
# 1. Provide a **summary** focused on key facts, named entities, quotes, or data points that are related to the topic.
# 2. Provide a **content_index** score between 0 and 10:

# ---
# Topic: {topic}
# Title: {title}
# [CONTENT START]
# {content}
# [CONTENT END]

# Respond in the following JSON format:
# {{
#   "summary": "<summary here>",
#   "content_index": <integer 0-10>
# }}
# """
#     try:
#         response = model.generate_content(prompt)
#         text = re.sub(r"^```(?:json)?", "", response.text.strip()).strip("```").strip()
#         return json.loads(text)
#     except Exception as e:
#         return {"summary": f"Error: {e}", "content_index": 0}

# def join_summaries(summaries, threshold=6):
#     return " ".join([
#         s['summary'].replace("\n", " ").replace("\r", " ")
#         for s in summaries if s.get("content_index", 0) > threshold
#     ])

# def extract_images_with_context(url):
#     driver.get(url)
#     time.sleep(3)
#     soup = BeautifulSoup(driver.page_source, "html.parser")
#     images_data = []
#     for img in soup.find_all("img"):
#         alt = img.get("alt", "")
#         src = img.get("src", "") or img.get("data-src", "")
#         context = ""
#         parent = img.find_parent(["figure", "div", "p"])
#         if parent:
#             context = parent.get_text(strip=True)
#         images_data.append({"alt": alt, "src": src, "context": context})
#     return images_data

# def generate_top_images_from_page(page, topic, model):
#     title = page['title']
#     url = page['url']
#     images = page['images']
#     images_text = "\n".join(
#         f"[{img['index']}] ALT: {img['alt']} | CONTEXT: {img['context']}"
#         for img in images
#     )
#     prompt = f"""
# You are an AI assistant selecting the most relevant images from a web page for the topic: "{topic}".

# Page Title: {title}
# Page URL: {url}

# [IMAGES START]
# {images_text}
# [IMAGES END]

# Your task:
# - Choose the **5 most relevant images** for the given topic.
# - For each image, return:
#   - The **index** of the image.
#   - A short **description** (1–2 sentences) that explains how it's related to the topic or why it's useful.

# Respond ONLY in this JSON format:
# [
#   {{
#     "index": <int>,
#     "description": "<description>"
#   }},
#   ...
# ]
# """
#     try:
#         response = model.generate_content(prompt)
#         text = re.sub(r"^```(?:json)?", "", response.text.strip()).strip("```").strip()
#         return {
#             "title": title,
#             "url": url,
#             "top_images": json.loads(text)
#         }
#     except Exception as e:
#         return {"title": title, "url": url, "top_images": [], "error": str(e)}

# def resolve_image_references(carousel_json, transformed_results):
#     for slide in carousel_json.values():
#         for element in slide.values():
#             content = element.get("content", "")
#             if content.startswith("image:"):
#                 try:
#                     page_idx, image_idx = map(int, content.replace("image:", "").split("-"))
#                     url = transformed_results[page_idx]['top_images'][image_idx]['src']
#                     element['content'] = url
#                 except:
#                     element['content'] = ""
#     return carousel_json

# # --- Main API Endpoint ---
# @app.route('/generate-carousel', methods=['POST'])
# def generate_carousel():
#     try:
#         data = request.get_json()
#         topic = data.get("topic")
#         json_template = data.get("json_template")

#         if not topic or not json_template:
#             return jsonify({"error": "Missing 'topic' or 'json_template'"}), 400

#         # Step 1: Search
#         search = GoogleSearch({
#             'q': topic,
#             'api_key': serp_api_key,
#             'engine': 'google',
#             'num': '10'
#         })
#         results = search.get_dict().get('organic_results', [])
#         summaries, links_only, pages_with_images = [], [], []

#         # Step 2: Scrape, summarize, extract images
#         for result in results:
#             title, link = result.get('title'), result.get('link')
#             if not title or not link:
#                 continue
#             content = scrape_and_summarize(link)
#             if content:
#                 links_only.append(link)
#                 summary = gemini_model_summary(title, content, topic)
#                 summaries.append({
#                     "topic": title,
#                     "summary": summary.get("summary", ""),
#                     "content_index": summary.get("content_index", 0)
#                 })
#                 images = extract_images_with_context(link)
#                 valid_images = [
#                     {
#                         "index": idx,
#                         "alt": img.get('alt', ''),
#                         "src": img.get('src', ''),
#                         "context": img.get('context', '')
#                     }
#                     for idx, img in enumerate(images)
#                     if img.get('src', '').lower().endswith(('.jpg', '.png', '.webp')) and (img.get('alt') or img.get('context'))
#                 ]
#                 pages_with_images.append({
#                     "title": title,
#                     "url": link,
#                     "images": valid_images
#                 })

#         # Step 3: Select top images
#         results_with_top_images = [
#             generate_top_images_from_page(page, topic, model)
#             for page in pages_with_images
#         ]
#         transformed_results = [
#             {
#                 "page_index": idx,
#                 "top_images": res.get("top_images", [])
#             } for idx, res in enumerate(results_with_top_images)
#         ]

#         # Step 4: Final Gemini Prompt
#         summary_text = join_summaries(summaries)
#         prompt = f"""
# You are an AI assistant designing carousel slides for the topic "{topic}".

# Summarized Content:
# \"\"\"
# {summary_text}
# \"\"\"

# Available Images:
# {json.dumps(transformed_results, indent=2)}

# JSON Template:
# {json.dumps(json_template, indent=2)}

# Follow these rules:
# - Fill "text" fields using summaries and tone guidance.
# - Replace "url" with image references like image:page_index-image_index
# - Keep JSON structure identical.

# Return ONLY the completed JSON.
# """
#         response = model.generate_content(prompt)
#         raw_output = re.sub(r"^```(?:json)?", "", response.text.strip()).strip("```").strip()
#         carousel_json = json.loads(raw_output)

#         # Step 5: Replace image refs with real URLs
#         resolved = resolve_image_references(carousel_json, results_with_top_images)
#         return jsonify({"carousel_json": resolved})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # --- Run the app ---
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=7860)


import os
import re
import json
import time
import requests

from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain.document_loaders import WebBaseLoader
from serpapi import GoogleSearch
import google.generativeai as genai
from google.generativeai import GenerativeModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from dotenv import load_dotenv

# --- App & Env Setup ---
app = Flask(__name__)
load_dotenv()

# --- API Keys ---
serp_api_key = os.environ.get("SERP_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=gemini_key)
model = GenerativeModel("gemini-2.0-flash-lite")

# --- Selenium Setup (headless for Render) ---
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=options)

# --- Helpers ---
def is_webpage(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return 'text/html' in response.headers.get('Content-Type', '').lower()
    except:
        return False

def clean_text(text, max_length=16000):
    return re.sub(r'\s+', ' ', text).strip()[:max_length]

def scrape_and_summarize(url, max_length=16000):
    if not is_webpage(url):
        return ""
    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()
    return clean_text(docs[0].page_content, max_length) if docs else ""

def gemini_model_summary(title, content, topic):
    prompt = f"""
You are an AI assistant analyzing article content to assess its relevance to a given topic and extract a high-quality summary.

Your tasks:
1. Provide a **summary** focused on key facts, named entities, quotes, or data points that are related to the topic.
2. Provide a **content_index** score between 0 and 10:

---
Topic: {topic}
Title: {title}
[CONTENT START]
{content}
[CONTENT END]

Respond in the following JSON format:
{{
  "summary": "<summary here>",
  "content_index": <integer 0-10>
}}
"""
    try:
        response = model.generate_content(prompt)
        text = re.sub(r"^```(?:json)?", "", response.text.strip()).strip("```").strip()
        return json.loads(text)
    except Exception as e:
        return {"summary": f"Error: {e}", "content_index": 0}

def extract_images_with_context(url):
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    images = []
    for img in soup.find_all("img"):
        src = img.get("src", "") or img.get("data-src", "")
        if not src:
            continue
        alt = img.get("alt", "")
        context = ""
        parent = img.find_parent(["figure", "div", "p"])
        if parent:
            context = parent.get_text(strip=True)
        images.append({"alt": alt, "src": src, "context": context})
    return images

def generate_top_images_from_page(page, topic):
    title = page['title']
    url = page['url']
    images = page['images']
    images_text = "\n".join(
        f"[{img['index']}] ALT: {img['alt']} | CONTEXT: {img['context']}" for img in images
    )
    prompt = f"""
You are an AI assistant selecting the most relevant images from a web page for the topic: "{topic}".

Page Title: {title}
Page URL: {url}

[IMAGES START]
{images_text}
[IMAGES END]

Your task:
- Choose the **5 most relevant images** for the given topic.
- For each image, return:
  - The **index** of the image.
  - A short **description** (1–2 sentences) that explains how it's related to the topic or why it's useful.

Respond ONLY in this JSON format:
[
  {{
    "index": <int>,
    "description": "<description>"
  }},
  ...
]
"""
    try:
        response = model.generate_content(prompt)
        text = re.sub(r"^```(?:json)?", "", response.text.strip()).strip("```").strip()
        return {
            "title": title,
            "url": url,
            "top_images": json.loads(text)
        }
    except Exception as e:
        return {"title": title, "url": url, "top_images": [], "error": str(e)}

def resolve_image_references(carousel_json, transformed_results):
    for slide in carousel_json.values():
        for element in slide.values():
            content = element.get("content", "")
            if content.startswith("image:"):
                try:
                    page_idx, image_idx = map(int, content.replace("image:", "").split("-"))
                    url = transformed_results[page_idx]['top_images'][image_idx]['src']
                    element['content'] = url
                except:
                    element['content'] = ""
    return carousel_json

def join_summaries(summaries, threshold=6):
    return " ".join([
        s['summary'].replace("\n", " ") for s in summaries
        if s.get("content_index", 0) > threshold
    ])

# --- API Route ---
@app.route('/generate-carousel', methods=['POST'])
def generate_carousel():
    try:
        data = request.get_json()
        topic = data.get("topic")
        json_template = data.get("json_template")

        if not topic or not json_template:
            return jsonify({"error": "Missing 'topic' or 'json_template'"}), 400

        # Step 1: Google Search via SerpAPI
        search = GoogleSearch({
            'q': topic,
            'api_key': serp_api_key,
            'engine': 'google',
            'num': '10'
        })
        results = search.get_dict().get('organic_results', [])

        summaries, pages_with_images = [], []

        for result in results:
            title = result.get('title')
            link = result.get('link')
            if not title or not link:
                continue
            content = scrape_and_summarize(link)
            if not content:
                continue

            summary = gemini_model_summary(title, content, topic)
            summaries.append({
                "topic": title,
                "summary": summary.get("summary", ""),
                "content_index": summary.get("content_index", 0)
            })

            images = extract_images_with_context(link)
            valid_images = [
                {
                    "index": idx,
                    "alt": img.get('alt', ''),
                    "src": img.get('src', ''),
                    "context": img.get('context', '')
                }
                for idx, img in enumerate(images)
                if img.get('src', '').lower().endswith(('.jpg', '.png', '.webp'))
                   and (img.get('alt') or img.get('context'))
            ]

            pages_with_images.append({
                "title": title,
                "url": link,
                "images": valid_images
            })

        results_with_top_images = [
            generate_top_images_from_page(page, topic) for page in pages_with_images
        ]

        transformed_results = [
            {
                "page_index": idx,
                "top_images": res.get("top_images", [])
            } for idx, res in enumerate(results_with_top_images)
        ]

        # Step 4: Final Prompt to Gemini
        summary_text = join_summaries(summaries)
        final_prompt = f"""
You are an AI assistant designing carousel slides for the topic "{topic}".

Summarized Content:
\"\"\"{summary_text}\"\"\"

Available Images:
{json.dumps(transformed_results, indent=2)}

JSON Template:
{json.dumps(json_template, indent=2)}

Follow these rules:
- Fill "text" fields using summaries and tone guidance.
- Replace "url" with image references like image:page_index-image_index
- Keep JSON structure identical.

Return ONLY the completed JSON.
"""
        response = model.generate_content(final_prompt)
        raw_output = re.sub(r"^```(?:json)?", "", response.text.strip()).strip("```").strip()
        carousel_json = json.loads(raw_output)

        # Step 5: Replace image references with actual URLs
        final_output = resolve_image_references(carousel_json, results_with_top_images)
        return jsonify({"carousel_json": final_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Health Check ---
@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running"}), 200

# --- Run App (Render Friendly) ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

