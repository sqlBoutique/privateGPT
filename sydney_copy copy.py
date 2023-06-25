from dotenv import load_dotenv
# from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
import os
import os.path
import argparse
from langchain import OpenAI
import streamlit as st
# import webbrowser
# import html 

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

if 'memory' not in st.session_state:
    st.session_state['memory'] = ''
if 'history' not in st.session_state:
    st.session_state['history'] = ''
if 'key' not in st.session_state:
    st.session_state['key'] = ''

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

#--------------------------------------------------------------

# Set the page and page configuration
st.set_page_config(
    page_title="Sydney Q&A App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!",}
)

# Sidebar contents
with st.sidebar:
    st.title('Sydney the Kidney 💬')
    st.markdown('''
    ## About
    This app is brought to you by:\n 
    [The Road Back To Life](https://kidneysupportgroup.org/)
    ''')

st.subheader("Ask Sydney 💬")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        footer:before {
            content:'Brought to you by: The Road Back To Life'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
        }
        .css-z5fcl4 {
            width: 100%;
            padding: 2.5rem 1rem 2rem;
            min-width: auto;
            max-width: initial;
        }
        .css-1544g2n {
            padding: 2rem 1rem 1rem;
}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

#--------------------------------------------------------------

def add_source(vsource):
    with source_expander:
        st.write(vsource)

        #   html_code = '[' + vsource + '](' + os.path.join(os.getcwd(), vsource) + ')\n'
        # #   encoded_code = html.escape(html_code)
        #   st.markdown(html_code)

        # st.button(vsource, on_click=webbrowser.open(os.path.join(os.getcwd(), vsource)))

        # if st.button(vsource):
        #     webbrowser.open(os.path.join(os.getcwd(), vsource))
        #  st.write(
        #     '[' + vsource + '](' +  os.path.join(os.getcwd(), vsource) + ')\n'
        # )  
# Parse the command line arguments
args = parse_arguments()

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

@st.experimental_singleton
def init_db():
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    return db

db = init_db()

@st.experimental_singleton
def init_retriever():
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    return retriever

retriever = init_retriever()

memory = ConversationBufferMemory()
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
# test that the API key exists
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    print("OPENAI_API_KEY is not set")
    exit(1)
else:
    print("OPENAI_API_KEY is set")

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)

# os.getenv("OPENAI_API_KEY") 
# llm = OpenAI(temperature=0)
llm = OpenAI(temperature=0.5)
# construct the base chain
# qa = RetrievalQA.from_chain_type(llm=llm, 
#                                  chain_type="stuff", 
#                                  retriever=retriever, 
#                                  memory=ConversationBufferMemory(), 
#                                  chain_type_kwargs=chain_type_kwargs,
#                                  return_source_documents=True
#                                  )

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
chain = load_qa_chain(llm=llm, chain_type="stuff")
container = st.container()
source_expander = st.expander("Sources:")

with st.form("main_form", clear_on_submit=True):
    
    query = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Submit")

    if submitted and query:
        # call the chain
        res = qa({"query": query})
        answer, docs = res['result'], res['source_documents']

        # answer = res['result']
        # docs = res['source_documents']
        # Get the answer from the chain

        with get_openai_callback() as cb:
            # response = qa.run(input_documents=docs, question=query)
            response = chain.run(input_documents=docs, question=query)
            print(cb)
            st.sidebar.write(cb)

        memory.save_context({"input":query}, {"output": response})

        print(query)
        # print(response)

        xsources = set()
        for document in docs:
            print('\n####----->', document.metadata["source"])
            print(document.page_content)
            xsources.add(document.metadata["source"]) 

        # Print the relevant result and sources used for the answer
        usource = ''
        vsources = ''
        zsources = ''
        container.write(query)    
        container.write(answer)


        for ysources in xsources:
            # container.write(os.path.basename(ysources))  
            zsources += '\n' + os.path.basename(ysources) + '\n'
            print(os.path.basename(ysources), ysources)
            print(os.getcwd())
            print(os.path.join(os.getcwd(), ysources))
            add_source(os.path.join(os.getcwd(), ysources))

            hst = ''

        with open('history.txt', 'a') as f:
            f.write(query + '\n\n')
            f.write(response + '\n\n')   
            f.write(zsources + '\n\n') 
            f.write('----------------------------------------------------\n\n') 

        if os.path.isfile('history.txt'):
            with open('history.txt', 'r') as f:
                hst = f.read()

        print(vsources)
        expander = st.expander("History")
        expander.write(hst)

