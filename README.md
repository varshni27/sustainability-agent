# 🌱 Emissions Analysis and Insights Agent
## Overview
### Retrieval-Augmented Generation (RAG) agent for sustainability insights

This project implements a Retrieval-Augmented Generation (RAG) agent that retrieves context from company reports, peer disclosures, and the Greenhouse Gas Protocol to generate grounded sustainability insights. The system uses Gemini 2.5 Flash as the reasoning engine, a Pinecone vector database for document retrieval, and a Streamlit application for the user interface.

## Technical Solution

The AI Agent consists of three major components:

**Document Ingestion:**
PDFs and CSVs are converted into embeddings and stored in a Pinecone vector database using deterministic IDs to prevent duplication. Semantic search enables fast and accurate retrieval from multiple documents at once.

**AI Agent:**
Gemini 2.5 Flash is used with advanced prompt engineering techniques such as ReAct, chain-of-thought prompting, few-shot examples, and the CO-STAR framework. Retrieved text and variables are injected dynamically into prompts to maintain relevance.

**Web Application:**
A Streamlit interface enables interactive queries, streams model responses to reduce latency, and exposes a temperature slider to balance creativity with deterministic outputs.
  <img width="1144" height="444" alt="image" src="https://github.com/user-attachments/assets/734267da-55b2-4e11-887a-0f8f45a88da3" />
## Installation

Clone the repository and install dependencies using: 
- pip install -r requirements.txt

## Environment Setup 
Create a .env file in the project root and add your API keys
- GOOGLE_API_KEY=your_google_api_key
- PINECONE_API_KEY=your_pinecone_api_key

## Running the Application
streamlit run app.py

- Enter a sustainability-related question in the text area and adjust the temperature slider to control the model’s creativity.

## Requirements

All dependencies are listed in requirements.txt.

## Appendix
- The one-pager delving into the design decisions and technical challenges can be found [here](https://www.notion.so/Emissions-Analysis-and-Insights-Agent-25fcadf03a6280dea9c2ca5898a8712b?source=copy_link)
 
