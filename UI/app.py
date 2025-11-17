from database.app_api import api_add,api_query,api_df_from_records,api_sentiment_distribution,api_classification_distribution
from database.db_layer import db_get_records

import streamlit as st
import pandas as pd
import whisper
import tempfile
from models.ai_api import ai_classifier,ai_ner,ai_stt
from pydub import AudioSegment
import unicodedata
import plotly.graph_objects as go
from table import FeedbackTableFormatter



# -------------------------
# ---- Streamlit UI ------
# -------------------------
def main():
    st.set_page_config(page_title="Patient Feedback Analyzer", layout="wide")
    if "whisper_model" not in st.session_state:
        st.session_state.whisper_model = whisper.load_model("small")

    NER_COLUMNS = ["doctor_name","staff_role","hospital_name","department","specialty","service_area","price","time_expression","location","quality_aspect","issue_type","treatment_type"]

    # Initialize session state stores
    if "feedback_text" not in st.session_state:
        st.session_state["feedback_text"] = ""
    if "ner_results" not in st.session_state:
        st.session_state["ner_results"] = {k: "" for k in NER_COLUMNS}
    if "update_stats" not in st.session_state:
        st.session_state["update_stats"] = False
    if "textbox_key" not in st.session_state:
        st.session_state.textbox_key = 0
    if "show_success" not in st.session_state:
        st.session_state.show_success = False
    if "ner_key_counter" not in st.session_state:
        st.session_state.ner_key_counter = 0

    # ---- Header with color and icon ----
    header_html = """
    <div style="background-color:#e74c3c;padding:20px;border-radius:10px;color:white;text-align:center;">
        <h1>🩺 Patient Feedback Analyzer</h1>
        <p style="font-size:16px;">Collect and analyze patient feedback efficiently</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.write("")  # spacing

    # ---- Main area split into two columns ----
    # Wrap columns in colored panels for more life
    col1_html = """
    <div style="background-color:#f0f8ff;padding:10px;border-radius:10px;box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
    """
    col2_html = """
    <div style="background-color:#fff0f5;padding:10px;border-radius:10px;box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
    """

    left_col, right_col = st.columns([1, 1], gap="small")
    with left_col:

        input_block = st.container()
        ner_block = st.container()
        stats_block = st.container()

        # -------------------------
        # LEFT COLUMN - stacked 3 equal parts
        # -------------------------
        with input_block:
                st.subheader("Input Feedback")
                input_left, input_right = st.columns([3, 1])

                # --- Microphone Controls ---
                with input_right:
                    st.markdown("**Microphone Controls**")
                    audio_value = st.audio_input("Record high quality audio", sample_rate=48000)

                    if st.button("Transcribe Audio") and audio_value is not None:
                        # Save temp WAV
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                            tmpfile.write(audio_value.getbuffer())
                            tmpfile_path = tmpfile.name

                        # Convert & transcribe
                        converted_path = tmpfile_path.replace(".wav", "_converted.wav")
                        sound = AudioSegment.from_file(tmpfile_path)
                        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                        sound.export(converted_path, format="wav")

                        # Transcribe
                        st.session_state.feedback_text = ai_stt(converted_path).strip()
                        st.success("Transcription complete!")

                        # Force rebuild of the text box
                        st.session_state.textbox_key += 1
                        st.rerun()

                # --- Text Box ---
                with input_left:
                    variable = st.session_state.get("feedback_text", "")
                    txt = st.text_area(
                        "Enter feedback (or record voice):",
                        value=variable,
                        height=180,
                        key=f"feedback_text_area_{st.session_state.textbox_key}"  # dynamic key each rerun
                    )
                    st.session_state.feedback_text = txt

                    # Add feedback button
                    if st.button("➕ Add Feedback"):
                        feedback_text = st.session_state.get("feedback_text", "").strip()
                        if not feedback_text:
                            st.error("No text to add. Type or record then transcribe.")
                        else:
                            # ✅ Call your backend API (sentiment + NER + DB insert)
                            api_add(feedback_text)
                            st.success("Feedback added successfully!")
                            st.rerun()

                        # Show success after rerun
                if st.session_state.show_success:
                    st.success("Transcription complete!")
                    st.session_state.show_success = False

                # ---- NER block (middle left) ----
                with ner_block:

                    st.subheader("NER Results")

                    # Create 3 columns for the 12 NER fields
                    cols = st.columns(3)

                    # --- Loop through each NER field ---

                    # --- Then safely render the fields ---
                    for i, key in enumerate(NER_COLUMNS):
                        col_index = i % 3
                        with cols[col_index]:
                            val = st.session_state.ner_results.get(key, "")
                            st.text_input(
                                label=key.replace("_", " ").title(),
                                value=val,
                                key=f"ner_{key}_{st.session_state.ner_key_counter}"
                            )

                    st.write("")  # spacing

                    # --- Run NER Button ---
                    run_ner_cols = st.columns([3, 1])
                    with run_ner_cols[1]:
                        if st.button("Run NER"):
                            feedback_text = st.session_state.get("feedback_text", "")
                            feedback_text = unicodedata.normalize("NFC", feedback_text).strip()
                            feedback_text = feedback_text.encode("utf-8", errors="ignore").decode("utf-8")

                            # Run NER
                            ner_result = ai_ner(feedback_text)
                            st.session_state.ner_results = ner_result

                            # Increment counter for dynamic keys (forces text inputs to refresh)
                            if "ner_key_counter" not in st.session_state:
                                st.session_state.ner_key_counter = 0
                            st.session_state.ner_key_counter += 1

                            print(f"The feedback Statement : {feedback_text}")
                            print(f"Has the following NER {ner_result}")

                            # Trigger rerun
                            st.rerun()

                # ---- Stats block (bottom left) ----
                with stats_block:
                    st.subheader("Statistics")
                    stat_col1, stat_col2 = st.columns(2)

                    # Fetch all records via API instead of direct DB call
                    if st.session_state.get("update_stats", False):
                        records = api_query([], [])  # fetch all records
                        st.session_state.update_stats = False  # reset flag
                    else:
                        records = api_query([], [])

                    df = api_df_from_records(records)

                    # Chart 1: sentiment distribution
                    with stat_col1:
                        # Let’s choose which sentiment column to visualize
                        sentiment_column = st.selectbox(
                            "Select sentiment category",
                            ["sentiment_pricing", "sentiment_appointments", "sentiment_staff",
                             "sentiment_customer_service", "sentiment_emergency_services"]
                        )

                        # Fetch counts from API
                        counts = api_sentiment_distribution(sentiment_column)

                        if counts:
                            fig = go.Figure(
                                data=[go.Pie(
                                    labels=[str(k) for k in counts.keys()],  # sentiment values 0,1,2
                                    values=list(counts.values()),
                                    hole=0.3  # optional: donut style
                                )]
                            )
                            fig.update_layout(title_text=f"{sentiment_column} Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No sentiment data to show.")

                    # Chart 2: classification distribution
                    with stat_col2:
                        # Choose classification category
                        class_column = st.selectbox(
                            "Select classification category",
                            ["sentiment_pricing", "sentiment_appointments", "sentiment_staff",
                             "sentiment_customer_service", "sentiment_emergency_services"]
                        )

                        # Get counts from API
                        counts = api_classification_distribution(class_column)

                        if counts:
                            fig = go.Figure(
                                data=[go.Pie(
                                    labels=[str(k) for k in counts.keys()],  # 0,1,2 sentiment values
                                    values=list(counts.values()),
                                    hole=0.3
                                )]
                            )
                            fig.update_layout(title_text=f"{class_column} Classification Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No classification data to show.")
    with right_col:
        # -------------------------
        # RIGHT COLUMN - Feedback records + filters
        # -------------------------
        with right_col:
            st.subheader("Feedback Records")

            # Filter panels (multi-select)
            st.markdown("### Filters")
            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                sentiment_options = ["Positive", "Negative", "Neutral"]
                selected_sentiments = st.multiselect(
                    "Sentiment", options=sentiment_options, default=sentiment_options
                )

            with filter_col2:
                classification_options = [
                    "Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"
                ]
                selected_classes = st.multiselect(
                    "Classification", options=classification_options, default=classification_options
                )
            refresh_col, download_col, clear_col = st.columns([1, 1, 1])

            with refresh_col:
                if st.button("Refresh"):
                    st.rerun()

            with download_col:
                if st.button("Download CSV"):
                    # Fetch all records via API
                    records_all = api_query([], [])
                    df_all = api_df_from_records(records_all)
                    if df_all.empty:
                        st.info("No data to download.")
                    else:
                        st.markdown(download_link_for_df(df_all), unsafe_allow_html=True)

            with clear_col:
                if st.button("Clear All"):
                    api_clear_feedback()
                    st.success("All records cleared.")
                    st.rerun()

            records = api_query(selected_sentiments, selected_classes)
            df_display = api_df_from_records(records)

            if df_display.empty:
                st.info("No records match the selected filters.")
            else:
                # Optional: format timestamp nicely
                if "timestamp" in df_display.columns:
                    df_display["timestamp"] = pd.to_datetime(df_display["timestamp"], errors="coerce")

                # Format and style the table
                formatter = FeedbackTableFormatter(df_display)
                styled_df = formatter.format_table()

                # Display in Streamlit
                st.dataframe(styled_df, height=500)


if __name__ == "__main__":
    main()
