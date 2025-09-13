import streamlit as st
import os
import tempfile
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from src.asr_whisper import transcribe_audio
from src.emotion_model import detect_emotion

st.set_page_config(
    page_title="Emotion-Aware ASR",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main .block-container {
            padding-top: 0 !important; ;
            padding-bottom: 30px;
        }
        
.fixed-header {
    position: fixed;
    top: 0; 
    left: 0; 
    width: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    z-index: 9999;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    padding: 5px 0; 
}

.header-container {
    display: flex;
    justify-content: center;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}
            
.nav-button {
    background: none;
    border: none;
    color: white;
    padding: 10px 20px; 
    margin: 0 6px;  
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.nav-button:hover {
    background-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-5px);
}

.content-wrapper {
    padding-top: 60px;
    max-width: 1200px;
    margin: 0 auto;
    padding: 70px 20px 30px;
}

.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
            
.emotion-card {
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
}

.transcript-box {
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

def main():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Home", use_container_width=True):
            st.session_state.current_page = "Home"
            st.rerun()
    
    with col2:
        if st.button("Get Started", use_container_width=True):
            st.session_state.current_page = "Get Started"
            st.rerun()
    
    with col3:
        if st.button("About", use_container_width=True):
            st.session_state.current_page = "About"
            st.rerun()
    
    if st.session_state.current_page == "Home":
        show_home()
    elif st.session_state.current_page == "About":
        show_about()
    elif st.session_state.current_page == "Get Started":
        show_start_using()

def show_home():
    st.title("Welcome to Emotion-Aware ASR")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        This Emotion-Aware Speech Recognition goes beyond just simple transcription. It main aim is to helps users understand the emotional context behind spoken words. Now understand not just what they say, but also how they say it in a single tap!
        
        #### How It Works:
        
        1. **You Share Audio**: Upload any audio recording of someone speaking
        2. **We Listen Carefully**: Our system transcribes the words and analyzes the speaker's tone
        3. **Double Analysis**: 
           - We detect emotions from the sound of the voice (tone, pitch, intensity)
           - We analyze the meaning of the words for emotional content
        4. **You Get Insights**: Receive a complete report showing both what was said and the emotional context
        
        Whether you're analyzing customer service calls, studying speech patterns, or just curious about emotional expression, 
        our tool helps you understand the full picture behind spoken communication.
        """)
        
        st.info(" **ⓘ Ready to understand voice emotions and tones?** Navigate to 'Get Started' to upload your first audio file!")
    
    with col2:
        with open("home-image.svg", "r") as f:
            svg_content = f.read()
        st.markdown(svg_content, unsafe_allow_html=True)


def show_about():
    st.title("Why Emotion-Aware ASR Matters")
    tab1, tab2, tab3 = st.tabs(["Purpose", "Technology", "Use Cases"])
    
    with tab1:
        st.markdown("""
        ### Our Mission
        
        In today's digital world, understanding not just *what* people say, but *how* they say it, 
        is crucial for meaningful communication. This Emotion-Aware ASR application bridges this gap 
        by providing comprehensive analysis of both speech content and emotional context.
        
        ### Why Emotion Matters
        
        - **Customer Service**: Understand customer satisfaction and frustration levels
        - **Healthcare**: Monitor patient emotional states during consultations
        - **Education**: Assess student engagement and comprehension
        - **Research**: Analyze emotional patterns in speech data
        - **Accessibility**: Provide richer context for hearing-impaired users
        """)
    
    with tab2:
        st.markdown("""
        ### Technical Architecture
        
        #### Speech Recognition
        - **Model**: OpenAI Whisper (small variant)
        - **Languages**: Text-based emotion analysis works best with English content. Audio-based emotion analysis may support multiple languages although would recommended english for this variant.
        
        #### Emotion Detection
        
        **Audio-based Analysis:**
        - Model: `superb/wav2vec2-base-superb-er`
        - Analyzes voice features, tone, and vocal patterns
        - Emotions: Angry, Happy, Neutral, Sad
        
        **Text-based Analysis:**
        - Model: `j-hartmann/emotion-english-distilroberta-base`
        - Analyzes semantic content and linguistic patterns
        - Emotions: Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
        
        #### Processing Pipeline
        1. Audio preprocessing and feature extraction
        2. Parallel ASR and audio emotion analysis
        3. Text emotion analysis on transcript
        4. Confidence scoring and result fusion
        5. Interactive visualization generation
        """)
    
    with tab3:
        st.markdown("""
        ### Real-World Applications
        
        #### Business & Customer Service
        - **Call Center Analytics**: Monitor agent performance and customer satisfaction
        - **Market Research**: Analyze emotional responses to products/services
        - **Sales Training**: Evaluate communication effectiveness
        
        #### Healthcare & Therapy
        - **Mental Health Assessment**: Track emotional patterns over time
        - **Patient Monitoring**: Detect distress or discomfort
        - **Therapy Sessions**: Analyze emotional progress
        
        #### Education & Training
        - **Student Engagement**: Measure attention and comprehension
        - **Language Learning**: Assess pronunciation and emotional expression
        - **Presentation Skills**: Provide feedback on delivery
        
        #### Research & Development
        - **Linguistic Studies**: Analyze emotion-speech correlations
        - **AI Training**: Generate labeled datasets
        - **User Experience**: Test emotional responses to interfaces
        """)

def show_start_using():
    st.title("Start Using Emotion-Aware ASR")    
    st.markdown("### Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV or MP3 format)",
        type=['wav', 'mp3'],
        help="Supported formats: WAV, MP3. The model automatically detects the language.")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name        
        st.success(f"File uploaded successfully")
        
        st.audio(uploaded_file.getvalue())
        
        if st.button("Analyze Audio", type="primary"):
            analyze_audio(temp_path)
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def analyze_audio(audio_path):
    with st.spinner("Processing audio... This may take a moment."):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Transcription
            status_text.text("Transcribing audio...")
            progress_bar.progress(25)
            transcript, detected_lang, whisper_lang = transcribe_audio(audio_path, model_size="small",force_language=None)
            
            # Step 2: Emotion Detection
            status_text.text("Analyzing emotions...")
            progress_bar.progress(75)
            
            (final_audio_label, audio_score, audio_predictions, 
             final_text_label, text_score, text_predictions) = detect_emotion(
                audio_path, transcript, debug=False)
            
            progress_bar.progress(100)
            display_results(
                transcript, detected_lang, whisper_lang,
                final_audio_label, audio_score, audio_predictions,
                final_text_label, text_score, text_predictions
            )
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.exception(e)

def display_results(transcript, detected_lang, whisper_lang, 
                   final_audio_label, audio_score, audio_predictions,
                   final_text_label, text_score, text_predictions):
    
    # Transcript section
    st.markdown("### Annotated Transcript")
    st.markdown(f"""
    <div class="transcript-box">
        <h4>"{transcript}"</h4>
        <p><strong>Emotion Annotation:</strong> [{final_audio_label} / {final_text_label}]</p>
        <p><strong>Language:</strong> {detected_lang or whisper_lang or 'Unknown'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Emotion Analysis
    st.markdown("### Emotion Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Audio-based Emotion")
        st.markdown(f"""
        <div class="emotion-card">
            <h3>{final_audio_label.title()}</h3>
            <p>Confidence: {audio_score:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio emotion chart
        if len(audio_predictions) > 1:
            audio_df = pd.DataFrame(audio_predictions, columns=['Emotion', 'Confidence'])
            fig_audio = px.bar(
                audio_df.head(4), 
                x='Confidence', 
                y='Emotion',
                orientation='h',
                title="Audio Emotion Confidence Scores",
                color='Confidence',
                color_continuous_scale='viridis')
            fig_audio.update_layout(height=300)
            st.plotly_chart(fig_audio, use_container_width=True)
    
    with col2:
        st.markdown("#### Text-based Emotion")
        st.markdown(f"""
        <div class="emotion-card">
            <h3>{final_text_label.title()}</h3>
            <p>Confidence: {text_score:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text emotion chart
        if len(text_predictions) > 1:
            text_df = pd.DataFrame(text_predictions[:6], columns=['Emotion', 'Confidence'])
            fig_text = px.bar(
                text_df, 
                x='Confidence', 
                y='Emotion',
                orientation='h',
                title="Text Emotion Confidence Scores",
                color='Confidence',
                color_continuous_scale='plasma')
            fig_text.update_layout(height=300)
            st.plotly_chart(fig_text, use_container_width=True)
    
    st.markdown("### Emotion Comparison")
    comparison_data = {
        'Analysis Type': ['Audio-based', 'Text-based'],
        'Primary Emotion': [final_audio_label.split('(')[0].strip(), final_text_label],
        'Confidence': [audio_score, text_score]}
    
    fig_comparison = px.bar(
        comparison_data,
        x='Analysis Type',
        y='Confidence',
        color='Primary Emotion',
        title="Audio vs Text Emotion Analysis Comparison",
        text='Primary Emotion')
    fig_comparison.update_traces(textposition='inside')
    fig_comparison.update_layout(height=400)
    st.plotly_chart(fig_comparison, use_container_width=True)

if __name__ == "__main__":
    main()