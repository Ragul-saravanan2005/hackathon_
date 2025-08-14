import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import re, os
import gradio as gr

# ----------------------------
# Load datasets
# ----------------------------
def load_nco_data():
    return pd.read_csv(r"C:\Users\ASUS\Downloads\penta\MOCK_DATA_with_NCO.csv")

def load_survey_data():
    ctl_file_path = r"C:\Users\ASUS\Downloads\penta\survey_data.ctl"
    with open(ctl_file_path, "r", encoding="utf-8") as f:
        ctl_content = f.read()

    infile_match = re.search(r"INFILE\s+'([^']+)'", ctl_content, re.IGNORECASE)
    if not infile_match:
        return pd.DataFrame()

    raw_path = infile_match.group(1).strip()
    if not os.path.isabs(raw_path):
        data_file_path = os.path.join(os.path.dirname(ctl_file_path), raw_path)
    else:
        data_file_path = raw_path

    if not os.path.exists(data_file_path):
        return pd.DataFrame()

    delimiter_match = re.search(r"FIELDS TERMINATED BY\s+'([^']+)'", ctl_content, re.IGNORECASE)
    delimiter = delimiter_match.group(1) if delimiter_match else ","

    try:
        df = pd.read_csv(data_file_path, delimiter=delimiter, encoding="utf-8", quotechar='"')
        return df
    except UnicodeDecodeError:
        return pd.read_csv(data_file_path, delimiter=delimiter, encoding="latin-1", quotechar='"')

# ----------------------------
# Load model + embeddings
# ----------------------------
nco_df = load_nco_data()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
nco_df['embeddings'] = nco_df['occupation_title'].apply(lambda x: model.encode(str(x), convert_to_tensor=True))

def search_occupation(query, top_k):
    query_emb = model.encode(query, convert_to_tensor=True)
    temp_df = nco_df.copy()
    temp_df['semantic_score'] = temp_df['embeddings'].apply(
        lambda emb: float(util.cos_sim(query_emb, emb))
    )
    temp_df['fuzzy_score'] = temp_df['occupation_title'].apply(
        lambda title: fuzz.token_sort_ratio(query.lower(), title.lower()) / 100
    )
    temp_df['final_score'] = 0.7 * temp_df['semantic_score'] + 0.3 * temp_df['fuzzy_score']
    results = temp_df.sort_values(by='final_score', ascending=False).head(top_k)
    return results[['occupation_title', 'nco_code', 'final_score']]

# ----------------------------
# Gradio Tab 1 - Survey Data
# ----------------------------
survey_df = load_survey_data()

def preview_survey():
    if survey_df.empty:
        return "⚠ Survey data not loaded.", None
    return f"Total records: {len(survey_df)}", survey_df.head(10)

def search_survey(column, value):
    if survey_df.empty:
        return "⚠ Survey data not loaded.", None
    if value.strip() == "":
        return "⚠ Please enter a search value", None
    results = survey_df[survey_df[column].astype(str).str.contains(value, case=False, na=False)]
    return f"Found {len(results)} records", results

# ----------------------------
# Build Gradio Interface
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 SurveyX (Gradio Version)")
    with gr.Tab("📊 Survey Data Explorer"):
        gr.Markdown("### Browse & Search Survey Data")
        show_btn = gr.Button("Preview Survey Data")
        preview_text = gr.Textbox(label="", interactive=False)
        preview_table = gr.Dataframe(interactive=False)
        show_btn.click(preview_survey, outputs=[preview_text, preview_table])

        gr.Markdown("#### Search in Survey Data")
        column_dd = gr.Dropdown(choices=survey_df.columns.tolist() if not survey_df.empty else [], label="Select column")
        value_tb = gr.Textbox(label="Enter search value")
        search_btn = gr.Button("Search")
        search_text = gr.Textbox(label="", interactive=False)
        search_table = gr.Dataframe(interactive=False)
        search_btn.click(search_survey, inputs=[column_dd, value_tb], outputs=[search_text, search_table])

    with gr.Tab("🔍 NCO Occupation Search"):
        gr.Markdown("### Multilingual Occupation Search")
        job_input = gr.Textbox(label="Enter job title", placeholder="e.g. Software Engineer / மென்பொருள் பொறியாளர்")
        topk_slider = gr.Slider(1, 10, value=3, step=1, label="Number of results")
        search_occu_btn = gr.Button("Search NCO")
        occu_results = gr.Dataframe(interactive=False)
        search_occu_btn.click(search_occupation, inputs=[job_input, topk_slider], outputs=occu_results)

demo.launch(debug=True, share=True)
