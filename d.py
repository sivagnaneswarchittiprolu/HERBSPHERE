import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import speech_recognition as sr
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import cv2
import tempfile

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Predefined Herbal PDFs
predefined_pdfs = ["Herbal_Plants.pdf", "plants.pdf"]

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

# Function to extract text using multiprocessing
def get_pdf_text(pdf_files):
    with multiprocessing.Pool() as pool:
        texts = pool.map(extract_text_from_pdf, pdf_files)
    return "\n".join(texts)

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to store text embeddings
@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Conversational Chain Setup
def get_conversational_chain():
    prompt_template = """
    Use the provided context to answer the question. If the context lacks information, say: 
    'The answer is not available in the context.' Don't guess.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply:", response["output_text"])

# Function to handle voice input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Error connecting to speech recognition service."

# Function to capture photo using the camera
def capture_photo():
    st.info("Please take a photo of the plant using your camera.")
    camera = cv2.VideoCapture(0)
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Display the frame in Streamlit
        st.image(frame, channels="BGR")
        
        # Capture photo on button press
        if st.button("Capture Photo"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file_name = tmp_file.name + ".jpg"
                cv2.imwrite(tmp_file_name, frame)
                st.image(tmp_file_name)
                camera.release()
                return tmp_file_name
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

# Streamlit App Main Function
def main():
    st.set_page_config("HERBSPHERE AI Chatbot")
    st.header("HERBSPHERE AI Chatbot 🌿")
    
    # Load or process FAISS index
    if not os.path.exists("faiss_index"):
        with st.spinner("Processing Herbal Documents..."):
            raw_text = get_pdf_text(predefined_pdfs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
    
    # Text Query Input
    user_question = st.text_input("Ask about herbal plants:")
    if user_question:
        user_input(user_question)
    
    # Voice Input Option
    if st.button("Use Voice Input"):
        voice_query = recognize_speech()
        if voice_query:
            user_input(voice_query)
    
    # Capture Photo Option
    if st.button("Take a Photo of a Plant"):
        photo_path = capture_photo()
        if photo_path:
            st.write(f"Photo captured: {photo_path}")
            # Here you can add further logic for plant identification if needed
    
    # Sidebar Daily Herbal Insight
    st.sidebar.header("Daily Herbal Insight 🌱")
    st.sidebar.write("Did you know? Tulsi is known for boosting immunity and reducing stress!")

if __name__ == "__main__":
    main()