import streamlit as st
from langchain_core.runnables import Runnable
from Utils.question_answering_RAG import (
    qa, init_llm_model, init_embeddings_model, create_vector_store, create_qa_model, init_prompt, gemini_generate_response, init_gemini_model
)
from Utils.summerization import summarize, summarize_pdf
from Utils.Image_captioning import query
from Utils.audio_input import listen
from Utils.audio_output import speak
from Utils.utils import read_file, read_text, read_pdf, read_csv, read_arxiv, read_markdown, get_file_extension 
from streamlit.runtime.uploaded_file_manager import UploadedFile
from PIL import Image

#===========================================Session State================================================
llm_model = init_llm_model()
embedding_model = init_embeddings_model()
prompt, qa_prompt = init_prompt()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_file" not in st.session_state:
    st.session_state.audio_file = ""

if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""

if "qa_model" not in st.session_state:
    st.session_state.qa_model = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

#==========================================Streamlit App=================================================
# Add logo
logo = Image.open("./documents/logo.png")
st.image(logo, width=500)

st.title("NismoGen")
st.write("### AI Chatbot with Multimodal Capabilities")
#=============================================Sidebar====================================================
# Display task selection
task_name = st.sidebar.selectbox(
    "Select the task you would like to perform:",
    ["Normal Chatbot", "Question Answering", "Text Summarization", "Image Captioning"],
)
# Display the repository link and authors information
st.sidebar.markdown(
    "## **[NismoGen Repo](https://github.com/youssefaymanmohamed/)**"
)
st.sidebar.markdown("## Done By:")
st.sidebar.markdown("##### **Youssef Ayman**")
st.sidebar.markdown(
    "##### [Github](https://github.com/youssefaymanmohamed) | [Email](mailto:youssefaymanmohamed1@gmail.com)"
)
st.sidebar.markdown("##### **Adham Ahmed**")
st.sidebar.markdown(
    "##### [Gitlab](https://github.com/adhamahmed46) | [Email](mailto:addham.taha@gmail.com)"
)

#===========================================Question Answering===========================================
if task_name == "Question Answering":
    uploaded_files = st.file_uploader(
        "Upload documents for context", type=["txt", "pdf", "csv"], accept_multiple_files=True
    )
    st.divider()

    # Select input method (text or audio)
    input_method = st.radio("Select input method:", ["Text", "Audio"])
    st.divider()

    # Handle audio input
    if input_method == "Audio":
        st.write("### Audio Input")
        if st.button("Record Audio"):
            recognized_text, audio_file_path = listen()
            st.session_state.audio_file = audio_file_path
            st.session_state.recognized_text = recognized_text
            st.write(f"Recognized text: **{recognized_text}**")
            st.audio(st.session_state.audio_file)

    # Process uploaded files
    if uploaded_files:
        st.write("Processing files...")
        st.session_state.vector_store = create_vector_store(uploaded_files, embedding_model) # Create a vector store from the uploaded files
        st.session_state.qa_model = create_qa_model(  # Create the QA model
            st.session_state.vector_store, llm_model, prompt, qa_prompt
        )
        st.write("Files processed successfully!")

    # Button to start a new chat
    if st.button("Start New Chat"):
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Input for user queries (Text and Audio both handled here)
    if prompt_text := st.chat_input(placeholder="Ask a question...") or st.session_state.recognized_text:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        st.chat_message("user", avatar="🧑‍💻").markdown(prompt_text)

        # Generate response
        if st.session_state.qa_model:
            response = ""
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Generating response..."):
                    generator = qa(prompt_text, st.session_state.qa_model, st.session_state.messages)
                for chunk in generator:
                    response += chunk
                st.write(response)
            
            # Append and play the response
            st.session_state.messages.append({"role": "assistant", "content": response})
            if st.button("Play Response Audio"):
                speak(response)

#==========================================Normal Chatbot================================================
elif task_name == "Normal Chatbot":
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = init_gemini_model()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Button to start a new chat
    if st.button("Start New Chat"):
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Add audio input button
    st.write("### Audio Input")
    if st.button("Record Audio"):
        recognized_text, audio_file_path = listen()
        st.session_state.audio_file = audio_file_path
        st.session_state.recognized_text = recognized_text
        st.write(f"Recognized text: **{recognized_text}**")
        st.audio(st.session_state.audio_file)

    # Input for user queries
    if prompt_text := st.chat_input(placeholder="Ask a question...") or st.session_state.recognized_text:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        st.chat_message("user", avatar="🧑‍💻").markdown(prompt_text)

        # Generate response
        response = ""
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Generating response..."):
                response = gemini_generate_response(prompt_text, st.session_state.gemini_model, st.session_state.messages)
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Play response audio
        if st.button("Play Response Audio"):
            speak(response)


#===========================================Text Summarization===========================================   
elif task_name == "Text Summarization":
    # Streamlit text area for user input or file uploader
    text = st.text_area("Enter text here:")
    
    uploaded_file = st.file_uploader(
        "Upload a file:", type=["pdf", "arxiv", "md", "docx", "json"]
    )

    # Handle uploaded files based on their extension
    if uploaded_file is not None:
        file_extension = get_file_extension(uploaded_file.name).lower()

        try:
            if file_extension == ".pdf":
                st.write("Processing PDF...")
                summary = summarize_pdf(uploaded_file)
                st.write("**Summary:**")
                st.write(summary)

                # Add Audio Output Button
                if st.button("Play Summary Audio"):
                    speak(summary)
            elif file_extension == ".csv":
                text = read_csv(uploaded_file)
            elif file_extension == ".arxiv":
                text = read_arxiv(uploaded_file)
            elif file_extension == ".md":
                text = read_markdown(uploaded_file)
            elif file_extension == ".docx":
                text = read_file(uploaded_file)  # Add a specific function if `docx` handling is unique
            elif file_extension == ".json":
                text = uploaded_file.read().decode("utf-8")  # Simple decoding for JSON files
            else:
                st.error("Unsupported file type. Please upload a valid file.")
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

    # Summarize the text
    if st.button("Summarize"):
        if text.strip():
            summary = summarize(text)
            st.write("**Summary:**")
            st.write(summary)

            # Add Audio Output Button
            if st.button("Play Summary Audio"):
                speak(summary)
        else:
            st.error("Please provide text or upload a file for summarization.")


#===========================================Image Captioning=============================================   
elif task_name == "Image Captioning":
    # Streamlit file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        caption = query(image)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, use_container_width=True, caption="Uploaded Image")
        with col2:
            st.write("**Caption:**")
            st.write(caption)
            
            # Add Audio Output Button
            if st.button("Play Caption Audio"):
                speak(caption)