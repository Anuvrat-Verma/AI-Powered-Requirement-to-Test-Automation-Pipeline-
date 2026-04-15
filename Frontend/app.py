import gradio as gr
import requests
import os

# ================== CONFIG ==================
BACKEND_URL = "http://127.0.0.1:8000"

# ===========================================

def upload_rag_files(files):
    if not files:
        return

    file_data = []
    for f in files:
        # Safely handle both filepaths (strings) and file objects based on Gradio version
        file_path = f if isinstance(f, str) else f.name
        file_data.append(("files", (os.path.basename(file_path), open(file_path, "rb"))))

    try:
        resp = requests.post(f"{BACKEND_URL}/upload_docs", files=file_data)
        if resp.status_code == 200:
            gr.Info(f"✅ Knowledge Base Updated: {resp.json().get('message')}")
        else:
            gr.Warning(f"❌ Upload failed: {resp.text}")
    except Exception as e:
        gr.Error(f"❌ Backend connection error: {str(e)}")


def process_requirements(text: str, audio_file, use_rag: bool = False):
    if not text and not audio_file:
        return "❌ Please enter text or record audio.", "", ""

    transcribed_text = text

    # Step 1: Transcribe audio if provided
    if audio_file:
        try:
            with open(audio_file, "rb") as f:
                files = {"audio": f}
                resp = requests.post(f"{BACKEND_URL}/transcribe", files=files, timeout=1000)

            if resp.status_code == 200:
                transcribed_text = resp.json().get("transcribed_text", text)
            else:
                return f"❌ Transcription failed: {resp.text}", "", ""
        except Exception as e:
            return f"❌ STT Error: {str(e)}", "", ""

    # Step 2: Generate stories, test cases & code
    try:
        payload = {
            "requirements": transcribed_text,
            "use_rag": use_rag
        }

        resp = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=1000)

        if resp.status_code == 200:
            data = resp.json()
            return (
                data.get("user_stories", "No output"),
                data.get("test_cases", "No output"),
                data.get("test_code", "No output")
            )
        else:
            return f"❌ Generation failed: {resp.text}", "", ""

    except requests.exceptions.Timeout:
        return "⏱️ Timeout: The agents are taking too long.", "", ""
    except Exception as e:
        return f"❌ Error connecting to backend: {str(e)}", "", ""


# ====================== GRADIO UI ======================
with gr.Blocks(title="AI Requirement to Test", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 AI Requirement-to-Test Automation Pipeline")
    gr.Markdown("**Backend:** FastAPI | **Frontend:** Gradio | **LLM:** Ollama")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📝 Business Requirements",
                placeholder="As a customer, I want to login and view my order history...",
                lines=8
            )

            audio_input = gr.Audio(
                label="🎤 Or Speak Your Requirements (Voice Input)",
                sources=["microphone", "upload"],
                type="filepath"
            )

            use_rag = gr.Checkbox(
                label="🔍 Use Company Knowledge Base (RAG)",
                value=False,
                interactive=True
            )

            rag_upload = gr.File(
                label="📁 Upload Context Docs (.txt)",
                file_count="multiple",
                file_types=[".txt"]
            )

            generate_btn = gr.Button("🚀 Generate Tests", variant="primary", size="large")

        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("1. 📋 Gherkin User Stories"):
                    stories_output = gr.Markdown(value="*Generated stories will appear here...*")
                with gr.Tab("2. ✅ Test Cases"):
                    tests_output = gr.Markdown(value="*Generated test cases will appear here...*")
                with gr.Tab("3. 💻 Selenium Test Code"):
                    code_output = gr.Code(
                        language="python",
                        show_label=True,
                        value="# Automation code will appear here..."
                    )

    # Button Actions
    generate_btn.click(
        fn=process_requirements,
        inputs=[text_input, audio_input, use_rag],
        outputs=[stories_output, tests_output, code_output]
    )

    rag_upload.upload(
        fn=upload_rag_files,
        inputs=[rag_upload],
        outputs=None
    )

    gr.Markdown("---\nMade with FastAPI + Gradio + Ollama")

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )