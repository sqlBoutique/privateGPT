from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import os
import os.path
import argparse
from langchain.llms import OpenAI
import streamlit as st
from PIL import Image
from constants import CHROMA_SETTINGS

load_dotenv()

# test that the API key exists
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    print("OPENAI_API_KEY is not set")
    exit(1)
else:
    print("OPENAI_API_KEY is set")
#---------------------------------------------------------------------
# OPENAI_API_TYPE = "azure"
# OPENAI_API_BASE = "https://openairbtl.openai.azure.com/"
# OPENAI_API_VERSION = "2023-03-15-preview"
# OPENAI_API_KEY = 'e41247cac2584232990d163d0b1070be'
# DEPLOYMENT_NAME = 'first'
# OpenAI.Engine = 'first'
# os.environ['OPENAI_API_TYPE'] = OPENAI_API_TYPE
# os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE
# os.environ['OPENAI_API_VERSION'] = OPENAI_API_VERSION
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# os.environ['DEPLOYMENT_NAME'] = DEPLOYMENT_NAME
#---------------------------------------------------------------------

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

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
image = Image.open('icons/basic_kidneyman1_icon_trans.png')
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

def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    # st.session_state.entity_memory.entity_store = {}
    # st.session_state.entity_memory.buffer.clear()

def add_source(vsource):
    with source_expander:
        st.write(vsource)

        #   html_code = '[' + vsource + '](' + os.path.join(os.getcwd(), vsource) + ')\n'
        # #   encoded_code = html.escape(html_code)
        #   st.markdown(html_code, unsafe_allow_html=True)

        # st.button(vsource, on_click=webbrowser.open(os.path.join(os.getcwd(), vsource)))

        # if st.button(vsource):
        #     webbrowser.open(os.path.join(os.getcwd(), vsource))
        #  st.write(
        #     '[' + vsource + '](' +  os.path.join(os.getcwd(), vsource) + ')\n'
        # )  
head_container = st.container()
with head_container:
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.image(image) 
    with col2:    
        st.subheader("Ask Sydney 💬")

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
# if "entity_memory" not in st.session_state:    
#     st.session_state.entity_memory.buffer = ""

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Parse the command line arguments
args = parse_arguments()

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
os.getenv("OPENAI_API_KEY") 
# llm = OpenAI(temperature=0)
llm = OpenAI(temperature=0.2)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
chain = load_qa_chain(llm=llm, chain_type="stuff")

container = st.container()
source_expander = st.expander("Sources:")

with st.form("main_form", clear_on_submit=True):
    
    query = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Submit") 

    if submitted and query:
        res = qa(query)
        # res = chain(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        # Get the answer from the chain
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
            st.sidebar.write(cb)

        st.session_state.past.append(query)
        st.session_state.generated.append(response)

        print(query)
        print(response)

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
        container.write(response)
        
        # container.write('[The Road Back To Life](https://kidneysupportgroup.org/)')
        # source_expander.write(zsources)
        # container.write('Sources: ')
       
        for ysources in xsources:
            # container.write(os.path.basename(ysources))  
            zsources += '\n' + os.path.basename(ysources) + '\n'
            print(os.path.basename(ysources), ysources)
            print(os.getcwd())
            print(os.path.join(os.getcwd(), ysources))
            add_source(os.path.join(os.getcwd(), ysources))

            # webbrowser.open(os.path.join(os.getcwd(), ysources))
            # os.startfile(os.path.join(os.getcwd(), ysources))
            # usource = '[' + ysources + '](' + os.path.join(os.getcwd(), ysources) + ')\n\n'
            # source_expander.write(usource)

            hst = ''

        with open('history.txt', 'a') as f:
            f.write(query + '\n\n')
            f.write(response + '\n\n')   
            f.write(zsources + '\n\n') 
            f.write('----------------------------------------------------\n\n') 

        if os.path.isfile('history.txt'):
            with open('history.txt', 'r') as f:
                hst = f.read()

        # source_expander = st.expander("Sources:")
        # source_expander.write(zsources)
        # source_expander.write(vsources)
        print(vsources)
        # expander = st.expander("History")
        # expander.write(hst)

# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])
        # st.info(st.session_state["past"][i],icon="🧐")
        # st.success(st.session_state["generated"][i], icon="🤖")

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session