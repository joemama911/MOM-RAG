import streamlit as st
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv
from pytube import YouTube
import os
import nomic

load_dotenv()

api_key=os.getenv("API_KEY")

nomic.cli.login(api_key)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

def video_audio_extraction(video_path):
    video = VideoFileClip(video_path)
    audio_file = "output_audio.mp3"
    audio = video.audio
    audio.write_audiofile(audio_file)
    return audio_file


def transcribe_audio_to_text(audio_file):
    model_size = "medium"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(audio_file, beam_size=5)
    
    transcription_file = "transcription.txt"
    with open(transcription_file, "w") as file:
        for segment in segments:
            file.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
    
    return transcription_file


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = NomicEmbeddings(
        model="nomic-embed-text-v1.5"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama3.1")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm( 
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain #here we are returning conversation_chain while its initiated in line 137 and this is the function where the prompt will be passed to 



def main():
    st.title("AI-Driven Meeting Minutes Generator")

    options= st.selectbox(
        "choose the required options for your input",
        options=("yt-link","video")

    )

    def y_tube(link):

        try:
            # Create YouTube object
            yt = YouTube(link)
            
            # Get the highest resolution stream available
            video_stream = yt.streams.get_highest_resolution()
            
            # Download the video
            print(f"Downloading: {yt.title}")
            video_stream.download(output_path="output.mp4")   
        except Exception as e:
            print(f"An error occurred: {e}")

    if options=="yt-link":

        link=st.text_input("drop your url here")

        uploaded_video=y_tube(link)
    else:
        uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        # Save uploaded video file temporarily
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("Video uploaded successfully!")
        
        if st.button("Process Video"):
            with st.spinner("Extracting audio and transcribing..."):
                # Step 1: Extract audio from video
                audio_file = video_audio_extraction(video_path)
                
                # Step 2: Transcribe audio to text
                transcription_file = transcribe_audio_to_text(audio_file)
                
                # Step 3: Read transcription text
                with open(transcription_file, "r") as file:
                    transcription_text = file.read()
                
                # Step 4: Split transcription into chunks
                text_chunks = get_text_chunks(transcription_text)
                
                # Step 5: Create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # Step 6: Create conversation chain
                st.session_state.conversation_chain = get_conversation_chain(vectorstore)
                
                st.success("Processing complete !")
                
    if "conversation_chain" in st.session_state:
            
            response = st.session_state.conversation_chain("summary of the context") #and here the prompt is only passed to a function called conversion_chain which is return while calling the get_conversation_chain
            st.write("*AI:*", response['answer'])
    

if __name__ == '_main_':
    main()
