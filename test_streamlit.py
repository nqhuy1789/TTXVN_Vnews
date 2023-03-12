import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json

DB_COLLECTION = 'TTXVN_Vnews_FAQs'
DB_COLLECTION_TITLE = 'TTXVN_Vnews_Title'

#Câu lệnh tìm kiếm theo vector
def Init():
    qdrant_client = QdrantClient(host='localhost', port=6333)
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    with open('data.json','r') as file:
        file_data = json.load(file)
    return qdrant_client, model, file_data

def Search_DB(qdrant_client, model, sQuery):
    embeddings = model.encode([sQuery])
    vector = embeddings[0].tolist()
    results = qdrant_client.search(
        collection_name=DB_COLLECTION,
        query_vector=vector,
        limit=5,
    )
    return results

def Search_DB_Title(qdrant_client, model, sQuery):
    embeddings = model.encode([sQuery])
    vector = embeddings[0].tolist()
    results = qdrant_client.search(
        collection_name=DB_COLLECTION_TITLE,
        query_vector=vector,
        limit=10,
    )
    return results


qdrant_client, model, file_data = Init()
option = st.selectbox('Chọn dữ liệu tìm kiếm?', ('Tiêu đề', 'FAQs'))
sQuery = st.text_input("Nội dung cần hỏi:", "")  
submitted1 = st.button(label = 'Tìm kiếm 🔎')

if submitted1 or sQuery:
    st.empty()
    if option=='FAQs':
        results = Search_DB(qdrant_client, model, sQuery)
        for result in results:
            key = str(result.payload['Key'])
            order = int(result.payload['Order'])
            YT_link = file_data['TTXVN'][key]['Liên kết']
            st.write('**Câu hỏi:** {}'.format(file_data['TTXVN'][key]['FAQ'][order][0]))
            st.write('**Câu trả lời [link]({}):** {} '.format(YT_link, file_data['TTXVN'][key]['FAQ'][order][1]))
    elif option=='Tiêu đề':
        results = Search_DB_Title(qdrant_client, model, sQuery)
        for result in results:
            key = str(result.payload['Key'])
            YT_link = file_data['TTXVN'][key]['Liên kết']
            st.write('**Tiêu đề:** {} [link]({})'.format(file_data['TTXVN'][key]['Tiêu đề'], YT_link))
