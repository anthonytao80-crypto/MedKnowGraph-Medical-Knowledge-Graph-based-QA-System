# MedKnowGraph —— Medical Knowledge Graph-based QA System

## 1.Introduction

<img src="./Medical_Knowledge_Graph_QA_System/graph.png" style="float:left"/>

This project presents a Graph-RAG based medical question-answering agent, which integrates knowledge graph reasoning, retrieval-augmented generation (RAG), and function calling to provide accurate and interpretable responses to medical questions.

The medical knowledge graph is constructed based on the open-source project by Liu Huanyong (Institute of Software, Chinese Academy of Sciences), which serves as the structured knowledge base. Building upon this foundation, the system employs a LangGraph-based architecture to orchestrate the workflow of question understanding, retrieval, reasoning, and response generation.

**Note:** This system is designed for **Chinese-language medical question answering**, supporting interaction entirely in Chinese.

To enhance answer quality, the agent incorporates several key components:

* Question rewriting

* Graph-based retrieval and scoring to identify relevant medical entities and relations

* Function calling

* LLM-driven response generation

## 2. Installation

### 2.1. Install Project Dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```

### 2.2 Docker Desktop Configuration

1. Install Docker Desktop
   Download and install Docker Desktop for your operating system.

2. Start PostgreSQL Service
   Locate the PostgreSQL Docker configuration file docker-compose.yml,
   then run the following command to start the PostgreSQL service in the background:
   `docker-compose up -d`
   After a successful startup, you can manage the container via Docker Desktop or through the command line.
   To remove the container and its volumes, run:

 	`docker-compose down --volumes`

3. Install pgvector Extension (Required by LangGraph's PostgresStore)
   Since LangGraph’s PostgresStore requires the pgvector extension, follow these steps inside the Docker container (you can do this directly in Docker Desktop):

   1. Install dependencies

```shell
apt update 

apt install -y git build-essential postgresql-server-dev-15 
```

2. Build and install pgvector

```shell
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git

cd pgvector

make

make install
```

3. Verify installation

```shell
ls -l /usr/share/postgresql/15/extension/vector\\\*
```

If you see the following file:

```
/usr/share/postgresql/15/extension/vector.control
```

it means the installation was successful

### 2.3 Dowload neo4j
Download and install neo4j for your operating system.

# 3. How to use
1. Build the medical knowledge graph
Run the following script to construct the medical knowledge graph in Neo4j:
```shell
python build_medicalgraph.py
```
2. Open the graph
Run the following command in terminal
```shell
neo4j.pat console
```
3. Make sure your docker is running

4. Chat via command line
You can directly interact with the agent through the terminal by running:
```shell
python ragAgent.py
```
5. Chat via user interface
Alternatively, run the following two scripts to start the web-based chat interface:
```shell
python main.py
python webUI.py
```
Once started, open the URL shown in the terminal (typically http://127.0.0.1:7860) to access the web interface and chat with the agent.


