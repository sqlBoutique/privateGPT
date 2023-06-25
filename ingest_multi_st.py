# C:\Users\Smoki\OneDrive - The Road Back To Life\Documents\Bob\Feed_the_Monster
# C:\Users\Smoki\OneDrive - The Road Back To Life\Documents\Bob\Monsters_been_Fed

#!/usr/bin/env python3
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS
import streamlit as st

load_dotenv()


#--------------------------------------------------------------

# # Set the page and page configuration
# st.set_page_config(
#     page_title="Sydney Import App",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Sidebar contents
# with st.sidebar:
#     st.title('Sydney the Kidney 💬')
#     st.markdown('''
#     ## About
#     This app is brought to you by:\n 
#     [The Road Back To Life](https://kidneysupportgroup.org/)
#     ''')

# st.subheader("import Sydney Documents 💬")

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         footer:before {
#             content:'Brought to you by: The Road Back To Life'; 
#             visibility: visible;
#             display: block;
#             position: relative;
#             #background-color: red;
#             padding: 5px;
#             top: 2px;
#         }
#         .css-z5fcl4 {
#             width: 100%;
#             padding: 2.5rem 1rem 2rem;
#             min-width: auto;
#             max-width: initial;
#         }
#         .css-1544g2n {
#             padding: 2rem 1rem 1rem;
# }
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

#--------------------------------------------------------------

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_documents = os.environ.get('SOURCE_DOCUMENTS', 'source_documents')
uploaded_documents = os.environ.get('UPLOADED_DOCUMENTS', 'uploaded_documents')
injested_documents = os.environ.get('INJESTED_DOCUMENTS', 'injested_documents')
# source_directory = os.environ.get('SOURCE_DIRECTORY', 'C:\\Users\\Smoki\\OneDrive - The Road Back To Life\\Documents\\Bob\\Monsters_been_Fed\\')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    ignored_files.clear() 
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_documents}")
    documents = load_documents(source_documents, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_documents}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")

    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # Set the page and page configuration
    st.set_page_config(
        page_title="Sydney Import App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar contents
    with st.sidebar:
        st.title('Sydney the Kidney 💬')
        st.markdown('''
        ## About
        This app is brought to you by:\n 
        [The Road Back To Life](https://kidneysupportgroup.org/)
        ''')

    st.subheader("import Sydney Documents 💬")
    st.subheader("Your documents")

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
    with st.form("my-form", clear_on_submit=True):
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.form_submit_button("Process"):
            for pdf_doc in pdf_docs:
                print(pdf_doc.name)
                with st.sidebar:
                    st.write(pdf_doc.name)
                    new_file = os.path.join(source_documents, pdf_doc.name)
                    with open(new_file, 'wb') as f:
                        f.write(pdf_doc.getvalue())

                    with st.spinner("Processing"):
                        if does_vectorstore_exist(persist_directory):
                            # Update and store locally vectorstore
                            print(f"Appending to existing vectorstore at {persist_directory}")
                            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
                            collection = db.get()
                            texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
                            print(f"Creating embeddings. May take some minutes...")
                            db.add_documents(texts)
                        else:
                            # Create and store locally vectorstore
                            print("Creating new vectorstore")
                            texts = process_documents()
                            print(f"Creating embeddings. May take some minutes...")
                            db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
                        db.persist()
                        db = None

                        print(f"Ingestion complete! You can now query your documents")



        # pdf_docs
        # for document in os.listdir(source_documents)
        #     print(os.path.basename(document))
        #     old_file = os.path.join(source_documents, document)
        #     print(os.path.abspath(old_file))
        #     xold_file = os.path.abspath(old_file).replace('\\', '/')
        #     # new_file = document.replace('source_documents', 'injested_documents')
        #     new_file = os.path.join(injested_documents, document)
        #     print(os.path.abspath(new_file))
        #     xnew_file = os.path.abspath(new_file).replace('\\', '/')
        #     os.rename(xold_file, xnew_file)


if __name__ == "__main__":
    main()
