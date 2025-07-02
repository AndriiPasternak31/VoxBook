import os
import json
import logging
import streamlit as st
from io import StringIO
from openai import OpenAI
from dotenv import load_dotenv
import whisper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import base64

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
load_dotenv(dotenv_path=".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables for embedding store
document_chunks = []
chunk_embeddings = []

st.header("VoxBook 📖🔊")
st.text(
    "Have a conversation with your favorite books from Project Gutenberg!\n"
    "Upload a .txt file, ask questions, and VoxBook will respond with both\n"
    "text and voice for a truly conversational reading experience."
)

# Voice settings
st.sidebar.header("🔊 Voice Settings")
voice_enabled = st.sidebar.checkbox("Enable Voice Responses", value=True, help="VoxBook will speak answers out loud")
voice_option = st.sidebar.selectbox(
    "Choose VoxBook's Voice:",
    ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    index=0,
    help="Different AI voices for VoxBook's responses"
)

if voice_enabled:
    st.sidebar.info("🎵 Voice responses are ON! VoxBook will speak answers out loud for a truly conversational experience.")
else:
    st.sidebar.info("🔇 Voice responses are OFF. You'll see text responses only.")

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tips:**")
st.sidebar.markdown("• Try different voices to find your favorite!")
st.sidebar.markdown("• Ask follow-up questions for deeper discussions")
st.sidebar.markdown("• Use audio or text input - both work great!")

def parse_uploaded_file(uploaded_file):
    """Parse uploaded .txt file into string and paragraphs."""
    if not uploaded_file:
        return "", []
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    paragraphs = []
    current_paragraph = []
    for line in string_data.splitlines():
        if line.strip() == "":
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(line.strip())
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))
    return string_data, paragraphs

def document_chunker(text, chunk_size=1000, overlap=200):
    """Split document into chunks for better retrieval."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            for i in range(end - 100, end):
                if i >= 0 and text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def build_embedding_store(chunks):
    """Build embedding store for document chunks."""
    global document_chunks, chunk_embeddings
    document_chunks = chunks
    
    # Create embeddings for chunks
    embeddings_response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    chunk_embeddings = [embedding.embedding for embedding in embeddings_response.data]
    
    return len(chunk_embeddings)

def paraphrase_question(question):
    """Generate paraphrases of the question."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Generate 2 paraphrased versions of the question. Return only the paraphrases, one per line."},
                {"role": "user", "content": question}
            ],
            max_tokens=100
        )
        content = response.choices[0].message.content
        if content:
            paraphrases = content.strip().split('\n')
            return [question] + [p.strip() for p in paraphrases if p.strip()]
        return [question]
    except:
        return [question]

def retrieve_relevant_chunks(question, top_k=3):
    """Retrieve most relevant chunks using similarity search."""
    if not chunk_embeddings or not document_chunks:
        return []
    
    # Generate paraphrases for better retrieval
    questions = paraphrase_question(question)
    
    # Embed all question variants
    question_embeddings_response = client.embeddings.create(
        input=questions,
        model="text-embedding-3-small"
    )
    question_embeddings = [embedding.embedding for embedding in question_embeddings_response.data]
    
    # Calculate similarities
    all_similarities = []
    for q_embedding in question_embeddings:
        similarities = cosine_similarity(np.array([q_embedding]), np.array(chunk_embeddings))[0]
        all_similarities.append(similarities)
    
    # Take max similarity across question variants
    max_similarities = np.max(all_similarities, axis=0)
    
    # Get top-k chunks
    top_indices = np.argsort(max_similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if max_similarities[idx] > 0.1:
            results.append((document_chunks[idx], float(max_similarities[idx])))
    
    return results

def analyze_question_with_context(question, relevant_chunks, book_title):
    """Use ChatGPT to analyze the question with relevant book context in a conversational way."""
    if not relevant_chunks:
        return "Hmm, I couldn't find relevant sections in the book to answer your question. Maybe try rephrasing it or asking about a different aspect of the story?"
    
    # Prepare context from relevant chunks
    context = "\n\n".join([chunk for chunk, _ in relevant_chunks])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You're VoxBook, a friendly and enthusiastic book discussion partner! 
                    You're having a casual conversation about '{book_title}' with someone who just asked a great question.
                    
                    Conversation style:
                    - Be warm and engaging, like talking to a friend at a book club
                    - Use conversational markers: "Oh, that's interesting!", "I love how...", "You know what I noticed..."
                    - Share insights like you're discovering them together
                    - Use "we" language: "Let's look at...", "We can see that..."
                    - Reference specific details from the text naturally
                    - Be enthusiastic about the book and the discussion
                    - If the excerpts don't fully answer the question, say so conversationally: "That's a great question! From what we can see here..."
                    
                    Keep your response engaging but concise (under 150 words)."""
                },
                {
                    "role": "user",
                    "content": f"I'm curious about this: {question}\n\nHere are some relevant parts from the book:\n\n{context}\n\n"
                              f"What do you think about this?"
                }
            ],
            max_tokens=400,
            temperature=0.8
        )
        
        content = response.choices[0].message.content
        return content.strip() if content else "Sorry, I'm having trouble putting together a response right now!"
    
    except Exception as e:
        logging.error(f"Error analyzing question: {e}")
        return f"Oops! I ran into a technical issue while thinking about your question: {str(e)}"

def generate_followup_questions(question, analysis, book_title):
    """Generate 2-3 follow-up questions based on current discussion."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""Based on this book discussion about '{book_title}', suggest 2-3 natural follow-up questions 
                    that would keep the conversation flowing. Make them:
                    - Conversational and curious: "What about...", "I'm curious about...", "Tell me more about..."
                    - Related to the current topic but exploring different angles
                    - Simple and engaging for any reader
                    - Short (under 15 words each)
                    
                    Return only the questions, one per line, without numbers."""
                },
                {
                    "role": "user",
                    "content": f"We just discussed: {question}\n\nAnd talked about: {analysis[:200]}...\n\nWhat would be good follow-up questions?"
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        if content:
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            return questions[:3]  # Return max 3 questions
        return []
    
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}")
        return []

def generate_questions(title):
    """Generate 5 simple discussion questions about the book."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Generate exactly 5 simple, accessible discussion questions about the book. 
                    Make them conversational and easy to understand, like you're talking to a friend about the book.
                    Focus on:
                    - Characters and their relationships
                    - What happens in the story
                    - Simple themes anyone can understand
                    - Personal reactions and feelings
                    
                    Use simple language. Start questions with words like "Tell me about...", "What do you think of...", "How does...", "Why did...", etc.
                    Return them as a numbered list (1., 2., 3., etc.)."""
                },
                {
                    "role": "user", 
                    "content": f"Generate 5 simple discussion questions about the book: {title}"
                }
            ],
            max_tokens=300,
            temperature=0.3  # Lower temperature for more consistent results
        )
        
        content = response.choices[0].message.content
        if not content:
            return ["Tell me about the main character in this book"]
        
        # Parse the questions from the response
        questions = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove number prefix and clean up
                question = line.split('.', 1)[-1].strip() if '.' in line else line.strip()
                if question.startswith(' '):
                    question = question[1:]
                if question:
                    questions.append(question)
        
        return questions[:5] if questions else ["Tell me about the main character in this book"]
        
    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        return ["Tell me about the main character in this book"]

def transcribe_audio(audio_value):
    """Transcribe user audio input using Whisper."""
    if not audio_value:
        return None
    
    # Save audio bytes to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_value.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        model = whisper.load_model("turbo")
        result = model.transcribe(tmp_file_path)
        return result['text']
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def embedder(input_data):
    """Embed questions, transcript, and paragraphs into a vector database."""
    embeddings = client.embeddings.create(
        input=input_data,
        model="text-embedding-3-small"
    )
    return embeddings

def text_to_speech(text, voice="alloy"):
    """Convert text to speech using OpenAI's TTS API."""
    try:
        # Clean up text for better TTS (remove markdown, excessive formatting)
        clean_text = text.replace("**", "").replace("*", "").replace("#", "")
        
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=clean_text
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            response.stream_to_file(tmp_file.name)
            return tmp_file.name
        
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        return None

def play_audio(audio_path):
    """Play audio file in Streamlit."""
    if audio_path and os.path.exists(audio_path):
        try:
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        except Exception as e:
            logging.error(f"Error playing audio: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except:
                pass

uploaded_file = st.file_uploader("Upload a file", type=["txt"], accept_multiple_files=False)
string_data, paragraphs = parse_uploaded_file(uploaded_file)

if string_data:
    logging.debug("string_data: %s", string_data[0:500])
    logging.info("File uploaded successfully")
    logging.debug("First paragraph: %s", paragraphs[0] if paragraphs else "No paragraphs found")
    logging.debug("Paragraphs length: %d", len(paragraphs))
    
    # Document Chunker & Embedding Store
    chunks = document_chunker(string_data)
    num_embeddings = build_embedding_store(chunks)
    success_msg = f"Great! I've processed your book into {len(chunks)} sections and I'm ready to discuss it with you!"
    st.info(f"📚 Document processed into {len(chunks)} sections")
    
    # Welcome voice message
    if voice_enabled:
        with st.spinner("🔊 VoxBook is welcoming you..."):
            audio_path = text_to_speech(success_msg, voice_option)
            if audio_path:
                st.write("🎵 **VoxBook says:**")
                play_audio(audio_path)

title = paragraphs[0].split("\n")[0] if paragraphs else "Unknown Title"

if string_data:
    st.write("**Discussion Questions:**")
    
    # Generate questions only once and store in session state
    if 'generated_questions' not in st.session_state or st.session_state.get('current_title') != title:
        st.session_state.generated_questions = generate_questions(title)
        st.session_state.current_title = title
    
    questions = st.session_state.generated_questions
    
    # Create clickable buttons for each question
    selected_question = None
    for i, q in enumerate(questions, start=1):
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            if st.button("📝", key=f"q_{i}_{hash(q)}", help="Click to explore this question"):
                selected_question = q
        with col2:
            st.markdown(f"**{i}.** {q}")
    
    # Store selected question in session state
    if selected_question:
        st.session_state.selected_question = selected_question

def process_question(question, title):
    """Process a question (either from audio or selected) and provide analysis."""
    if not question or not document_chunks:
        return
        
    st.write("💬 **Your Question:**")
    st.write(f"*{question}*")
    
    # Retriever - Find relevant chunks
    relevant_chunks = retrieve_relevant_chunks(question)
    
    if relevant_chunks:
        # Get ChatGPT analysis
        st.write("🤖 **VoxBook's Response:**")
        with st.spinner("Let me think about this..."):
            analysis = analyze_question_with_context(question, relevant_chunks, title)
            st.write(analysis)
            
            # Add voice response if enabled
            if voice_enabled and analysis:
                with st.spinner("🔊 VoxBook is speaking..."):
                    audio_path = text_to_speech(analysis, voice_option)
                    if audio_path:
                        st.write("🎵 **Listen to VoxBook's response:**")
                        play_audio(audio_path)
                    else:
                        st.warning("🔇 Voice generation failed, but you can read the response above!")
        
        # Generate and display follow-up questions
        st.write("---")
        st.write("🤔 **Keep the conversation going:**")
        
        with st.spinner("Thinking of follow-up questions..."):
            followup_questions = generate_followup_questions(question, analysis, title)
        
        if followup_questions:
            # Create buttons for follow-up questions
            cols = st.columns(len(followup_questions))
            for i, followup in enumerate(followup_questions):
                with cols[i]:
                    if st.button(f"💭 {followup}", key=f"followup_{hash(question)}_{i}", help="Click to explore this question"):
                        st.session_state.followup_question = followup
                        st.rerun()
        
        st.write("---")
        st.write("📖 **Supporting Evidence from the Book:**")
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            with st.expander(f"📄 Relevant Section {i} (Relevance: {score:.3f})"):
                st.write(chunk)
    else:
        no_results_msg = "Hmm, I couldn't find highly relevant sections for that question. Try asking about something else or rephrasing it!"
        st.write(f"❌ {no_results_msg}")
        
        # Voice the "no results" message too
        if voice_enabled:
            with st.spinner("🔊 VoxBook is speaking..."):
                audio_path = text_to_speech(no_results_msg, voice_option)
                if audio_path:
                    play_audio(audio_path)

# Handle follow-up questions from buttons
if hasattr(st.session_state, 'followup_question') and st.session_state.followup_question:
    st.write("### �� Follow-up Question")
    process_question(st.session_state.followup_question, title)
    # Clear the follow-up question after processing
    del st.session_state.followup_question

# Handle selected questions from buttons
if hasattr(st.session_state, 'selected_question') and st.session_state.selected_question:
    st.write("### 📝 Selected Question Analysis")
    process_question(st.session_state.selected_question, title)
    # Clear the selected question after processing
    del st.session_state.selected_question

# Quick question input
st.write("---")
st.write("💬 **Continue the conversation:**")
col1, col2 = st.columns([3, 1])
with col1:
    quick_question = st.text_input("Ask another question about the book...", placeholder="e.g., What happens to Winston at the end?", key="quick_question")
with col2:
    st.write("")  # Add some spacing
    ask_button = st.button("Ask!", key="ask_button")

if (quick_question and ask_button) or (quick_question and st.session_state.get('last_quick_question') != quick_question):
    if quick_question.strip():
        st.session_state.last_quick_question = quick_question
        st.write("### ❓ Your Question")
        process_question(quick_question, title)

# Handle audio input
audio_value = st.audio_input("🎤 Or speak your question:")
if audio_value:
    st.audio(audio_value)
    logging.info("Audio recorded successfully")
    transcription = transcribe_audio(audio_value)
    logging.info("Transcription result: %s", transcription)
    
    if transcription:
        st.write("### 🎤 Audio Question")
        process_question(transcription, title)
