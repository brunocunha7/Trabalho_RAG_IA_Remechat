import streamlit as st
from typing import List
import logging
import os
import pickle
import re
import numpy as np
import faiss
from dotenv import load_dotenv
import PyPDF2
from openai import OpenAI


def extract_text_from_pdf(pdf_path: str) -> str:
    logging.info(f"Extraindo texto do PDF: {pdf_path}")
    text = ''
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo PDF: {e}")
    return text

def split_text_into_chunks(text: str, max_chunk_size: int = 5000) -> List[str]:
    logging.info("Dividindo o texto em chunks.")
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    logging.info(f"Total de chunks criados: {len(chunks)}")
    return chunks

def get_embedding(text: str, client, model: str = "text-embedding-3-small") -> List[float]:
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"Erro ao obter embedding para o texto: {e}")
        return []

def get_embeddings(texts: List[str], client, model: str = "text-embedding-3-small") -> List[List[float]]:
    embeddings = []
    logging.info("Gerando embeddings para os chunks.")
    for i, text in enumerate(texts):
        embedding = get_embedding(text, client, model)
        embeddings.append(embedding)
        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            logging.info(f"Processados {i + 1}/{len(texts)} chunks.")
    return embeddings

def create_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    logging.info("Criando índice FAISS.")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def save_embeddings(embeddings: List[List[float]], chunks: List[str], index: faiss.IndexFlatL2,
                    embeddings_file: str = 'embeddings.pkl',
                    chunks_file: str = 'chunks.pkl',
                    index_file: str = 'faiss.index'):
    logging.info("Salvando embeddings, chunks e índice no disco.")
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, index_file)

def load_embeddings(embeddings_file: str = 'embeddings.pkl',
                    chunks_file: str = 'chunks.pkl',
                    index_file: str = 'faiss.index'):
    if os.path.exists(embeddings_file) and os.path.exists(chunks_file) and os.path.exists(index_file):
        logging.info("Carregando embeddings, chunks e índice do disco.")
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        index = faiss.read_index(index_file)
        return embeddings, chunks, index
    else:
        logging.warning("Arquivos de embeddings não encontrados.")
        return None, None, None

def search_index(index: faiss.IndexFlatL2, query_embedding: List[float], k: int = 5):
    logging.info("Pesquisando no índice FAISS por embeddings similares.")
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]

def answer_query(query: str, index: faiss.IndexFlatL2, chunks: List[str], client, k: int = 5) -> str:
    logging.info("Respondendo à pergunta do usuário.")
    query_embedding = get_embedding(query, client)
    indices, distances = search_index(index, query_embedding, k)
    relevant_chunks = [chunks[i] for i in indices]
    context = '\n\n'.join(relevant_chunks)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um médico farmaceutico que ajuda com perguntas sobre Remédios para diversos tipos de tratamento."},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        logging.error(f"Erro ao gerar a resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."

# Configuração principal
def main():
    # Configuração de logging e variáveis de ambiente
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("Chave da API da OpenAI não encontrada. Defina OPENAI_API_KEY no seu arquivo .env.")
        st.error("Erro: Chave da API da OpenAI não configurada.")
        return

    # Inicialização do cliente OpenAI
    client = OpenAI(api_key=api_key, max_retries=5)

    # Carregar ou criar embeddings
    embeddings, chunks, index = load_embeddings()
    if embeddings is None:
        st.warning("Embeddings não encontrados. Processando PDFs...")
        all_text = ""
        for file_name in os.listdir('bulas/'):
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join('bulas/', file_name)
                text = extract_text_from_pdf(pdf_path)
                all_text += text + "\n"

        chunks = split_text_into_chunks(all_text)
        embeddings = get_embeddings(chunks, client)
        index = create_faiss_index(embeddings)
        save_embeddings(embeddings, chunks, index)
        st.success("Embeddings criados e salvos com sucesso.")


    st.title("💬 Chat GPT + Bulário Anvisa")
    st.write("Faça perguntas sobre remédios e tratamentos, e eu responderei com base no contexto fornecido.")
    
    with st.sidebar:
        st.title("💊 ReméCHAT")
        st.subheader("Bulário Anvisa")
        st.markdown(
            """O RemeChat utiliza todas as bulas de remédios disponíveis no Bulário eletrônico do site da [Anvisa](https://consultas.anvisa.gov.br/#/bulario/)"""
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    

    # Entrada de texto do usuário
    user_input = st.text_input("Digite sua pergunta:")

    if st.button("Enviar"):
        if user_input.strip():
            # Adicionar ao histórico
            st.session_state.chat_history.append(("Você", user_input))

            # Responder à pergunta
            answer = answer_query(user_input, index, chunks, client)
            st.session_state.chat_history.append(("Assistente", answer))


    


    

    # Exibir histórico de conversa
    st.write("### Histórico de Conversa")
    for sender, message in st.session_state.chat_history:
        if sender == "Você":
            st.markdown(f"\n\n\n**{sender}:** {message}")
        else:
            st.markdown(f"*{sender}:* {message}")

if __name__ == "__main__":
    main()
