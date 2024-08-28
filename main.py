import streamlit as st
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


def video_audio_extraction(video_path):
    video = VideoFileClip(video_path)
    audio_file = "output_audio.mp3"
    audio = video.audio
    audio.write_audiofile(audio_file)
    return audio_file


def transcribe_audio_to_text(audio_file):
    model_size = "medium-v3"
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
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    st.title("AI-Driven Meeting Minutes Generator")

    uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        # Save uploaded video file temporarily
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("Video uploaded successfully!")
        
        if st.button("Process Video"):
            with st.spinner("Extracting audio and transcribing..."):

                audio_file = video_audio_extraction(video_path)
                

                transcription_file = transcribe_audio_to_text(audio_file)
                

                with open(transcription_file, "r") as file:
                    transcription_text = file.read()
                

                text_chunks = get_text_chunks(transcription_text)
                

                vectorstore = get_vectorstore(text_chunks)
                

                st.session_state.conversation_chain = get_conversation_chain(vectorstore)
                
                st.success("Processing complete! You can now ask questions based on the video content.")
                
    if "conversation_chain" in st.session_state:
        st.header("Ask Your Questions")
        
        if True:
            response = st.session_state.conversation_chain({"generate a nice minutes of meeting "})
            st.write("**AI:**", response['answer'])

if __name__ == '__main__':
    main()
