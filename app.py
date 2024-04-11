import streamlit as st
import pandas as pd
import base64
import os 
import re
import vertexai
import vertexai.preview.generative_models as generative_models

from langcodes import Language
from google.cloud import texttospeech
from vertexai.generative_models import GenerativeModel

# os.environ["GCLOUD_PROJECT"] = ""

path = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = path + "/static/"
MEDIA_DIR = path + "/media/"
KBC_LOGO = path + "/static/keboola_mini.png"
GC_LOGO = path + "/static/google_mini.png"
GEMINI_LOGO = path + "/static/gemini.png"

client = texttospeech.TextToSpeechClient()

def handle_change():
    """
    Set and update session state variables.
    """
    if st.session_state.input_type_choice:
        st.session_state.input_type = st.session_state.input_type_choice
    if st.session_state.input_text:
        st.session_state.text = st.session_state.input_text
    if st.session_state.file_type_choice:
        st.session_state.file_type = st.session_state.file_type_choice
    if st.session_state.voice_lang_choice:
        st.session_state.voice_lang = st.session_state.voice_lang_choice
    if st.session_state.voice_type_choice:
        st.session_state.voice_type = st.session_state.voice_type_choice
    if st.session_state.voice_name_choice:
        st.session_state.voice_name = st.session_state.voice_name_choice
    if st.session_state.audio_profile_choice:
        st.session_state.audio_profile = st.session_state.audio_profile_choice
    if st.session_state.voice_speed_choice:
        st.session_state.voice_speed = st.session_state.voice_speed_choice
    if st.session_state.voice_pitch_choice:
        st.session_state.voice_pitch = st.session_state.voice_pitch_choice

        
@st.cache_data()
def get_available_voices() -> pd.DataFrame:
    """
    Fetch available voices and return as a DataFrame.
    """
    try: 
        voices = client.list_voices()
        data = [{
            "name": voice.name, 
            "language_code": voice.language_codes[0], 
            "ssml_gender": texttospeech.SsmlVoiceGender(voice.ssml_gender).name, 
            "language": Language.get(voice.language_codes[0]).display_name(voice.language_codes[0]).title()
        } for voice in voices.voices]
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()


@st.cache_data()
def convert(in_text: str, in_file_type: str, out_file: str, lang_code: str, 
            lang_name: str, ssml_gender: texttospeech.SsmlVoiceGender, speaking_rate: float, 
            pitch: float, effects_profile_id: str) -> bool:
    """
    Synthesizes speech from the input string of text or SSML.
    Note: SSML must be well-formed according to:
        https://www.w3.org/TR/speech-synthesis/
    Turns the result into an audio file and stores it in the folder 'media'.
    Returns True on success.
    """
    synthesis_input = texttospeech.SynthesisInput(ssml=in_text) if in_file_type == "SSML" else texttospeech.SynthesisInput(text=in_text)
    voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=lang_name, ssml_gender=ssml_gender)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3 if out_file.endswith(".mp3") else texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=speaking_rate,
        pitch=pitch,
        effects_profile_id=[effects_profile_id] if effects_profile_id else []
    ) 
    try:        
        response = client.synthesize_speech(
            input=synthesis_input, 
            voice=voice, 
            audio_config=audio_config
        )
        with open(out_file, "wb") as out:
            out.write(response.audio_content)            
        return True
    except Exception as e:
        st.error("Error in converting text to speech: " + str(e))
        return False
    

def generate(content):
    vertexai.init(project="keboola-ai", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    #model = GenerativeModel("gemini-1.5-pro-latest")

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    
    responses = model.generate_content(
        contents=f"Create a 1-minute long introduction speech to welcome the audience at GCP Data Cloud Live, which is 1-day event in Bucharest. Return only the speech, the language should be English. Here's the input: {content}",
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    st.session_state.speech = "".join(response.text for response in responses)
    return st.session_state.speech

def get_speech():
    with st.form("my_form"):
        left_mid_col, mid_mid_col, right_mid_col = st.columns([1, 10, 1], gap="small")

        left_mid_col.image(GEMINI_LOGO)
        input = mid_mid_col.text_input("Generate intro speech with Gemini:")
        right_mid_col.write("####")
        if right_mid_col.form_submit_button("Submit", use_container_width=True):
            output = generate(input)
            #st.write(output)

def app():
    st.set_page_config(layout='wide')
    
    if 'speech' not in st.session_state:
        st.session_state.speech = ""
    
    # Logos
    logo_html = f'''
    <div style="display: flex; align-items: center; justify-content: right; font-size: 30px;">
        <img src="data:image/png;base64,{base64.b64encode(open(KBC_LOGO, "rb").read()).decode()}" style="height: 60px;">
        <span style="margin: 0 20px;">➕</span>
        <img src="data:image/png;base64,{base64.b64encode(open(GC_LOGO, "rb").read()).decode()}" style="height: 60px;">
        <span style="margin: 0 20px;">🟰</span>
        <span>❤️</span>
    </div>
    '''

    # Title & app desc
    st.markdown(f"{logo_html}", unsafe_allow_html=True)
    st.write("#####")
    st.header("Text-to-Speech 🤖 WIP")
    st.write("Convert plain text or SSML text into a WAV or MP3 audio file using Google's Cloud Text-to-Speech API – simply enter your plain or SSML text, choose from the various customization options, and then convert your text to an audio file.")
    
    alert_box = st.empty()
    st.markdown("---")  

    # Intro generation with Gemini 
    get_speech()
  
    # TEXT-TO-SPEECH SECTION
    input_type = st.radio(label="input_type", options=["Text", "SSML"], index=0, key="input_type_choice", label_visibility="hidden", horizontal=True)
    text = st.text_area("Text to speak:", placeholder=f"Enter {input_type} here", value=st.session_state.speech, key="input_text", height=250)

    # Get available voices
    voices = get_available_voices()
    if voices.size == 0:
        alert_box.error("Could not retrieve a list of available voices from the Google API!  Do not proceed.")
        st.stop()

    # Get the language names for the Language/locale dropdown
    language_names = voices["language"].sort_values().unique()

    # Get audio device profiles from JSON file
    with open(STATIC_DIR + "audio_profile_id.json") as f:
        profile_names = pd.read_json(f)

    # PARAMETERS
    left_col, mid_col, right_col = st.columns(3) 
    
    with left_col:
        # Language/locale dropdown
        selected_lang = st.selectbox("Language/locale:", on_change=handle_change, options=language_names, index=9 ,key="voice_lang_choice")
        
        # Update session state for selected language
        st.session_state.voice_lang = selected_lang

        # Filter voices by selected language
        filtered_voices = voices[voices["language"] == selected_lang]

        # Find Voice types
        pattern = r'(\w+)-[A-Z\d]$'
        filtered_voices.loc[:, 'voice_type'] = filtered_voices['name'].apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else None)
        voice_types = filtered_voices['voice_type'].unique().tolist()

        # Audio profile dropdown
        selected_profile = st.selectbox("Audio device profile:", on_change=handle_change, options=profile_names["name"] ,key="audio_profile_choice")

        # Output file format
        out_file_type_choice = st.radio("Choose the output file type:", on_change=handle_change, options=["WAV", "MP3"], key="file_type_choice", horizontal=True)
        out_file_name = f"audio.{out_file_type_choice.lower()}"
        out_file = str(MEDIA_DIR + out_file_name)
    
    with mid_col:
        # Voice type dropdown
        selected_voice_type = st.selectbox("Voice type:", on_change=handle_change, options=voice_types, key="voice_type_choice")
        
        # Voice speed slider
        selected_speed = st.slider("Speed:", min_value=0.25, max_value=4.00, value=1.00, key="voice_speed_choice")
    
    with right_col:
        # Filter voices by selected voice type
        final_filtered_voices = filtered_voices[filtered_voices['voice_type'] == selected_voice_type]
        final_filtered_voices = final_filtered_voices.copy()

        # Friendly names for dropdown
        final_filtered_voices["friendly_name"] = final_filtered_voices["name"].astype(str) + " (" + final_filtered_voices["ssml_gender"] + ")"
        # Voice name dropdown
        selected_voice = st.selectbox("Voice name:", on_change=handle_change, options=final_filtered_voices["friendly_name"], key="voice_name_choice")

        # Voice pitch slider 
        selected_pitch = st.slider("Pitch:", min_value=-20.00, max_value=20.00, value=0.0, step=0.1, key="voice_pitch_choice")
    
    # Add convert button 
    st.write("#####")
    send_button = st.button("Convert")
    st.markdown("---")
    
    # Only continue with the remaining script, if some plain text or SSML text has been provided
    if text:
        voice = final_filtered_voices[final_filtered_voices["friendly_name"]==selected_voice]
        profile = profile_names[profile_names["name"]==selected_profile]
        
        # Convert Text-to-Speech using all the collected arguments
        if send_button:
            result = convert(
                in_text=text,
                in_file_type=input_type, 
                out_file=out_file, 
                lang_code=voice["language_code"].values[0], 
                lang_name=voice["name"].values[0], 
                ssml_gender=voice["ssml_gender"].values[0],
                speaking_rate=selected_speed,
                pitch=selected_pitch,
                effects_profile_id=profile["id"].values[0]
            )
            
            # Only continue if the 'convert' function returned True
            if result:
                st.success("Audio file created successfully! You can now play or download your audio file. 🕺🏻")
                # Open/read the newly created audio.wav or audio.mp3 file
                with open(out_file, "rb") as file:
                    audio_bytes = file.read()
                    audio_format = "audio/wav" if "WAV" in out_file_type_choice else "audio/mpeg"
                    
                    # Display an audio player to play back newly created audio file
                    left_top_col, right_top_col = st.columns([8,1])

                    left_top_col.audio(audio_bytes, format=audio_format)
                
                    # Create a download button to download newly created audio file
                    right_top_col.download_button(
                            label="Download",
                            data=file,
                            file_name=out_file_name, 
                            use_container_width=True
                        )
            elif not result:
                st.error("Something went wrong! Could not convert the text using the Google API!")
                st.stop()


if __name__ == "__main__":
    app()