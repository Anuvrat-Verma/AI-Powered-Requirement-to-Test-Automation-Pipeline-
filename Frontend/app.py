import gradio as gr
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"


def process_requirements(text: str, audio_file, use_rag: bool = True):
    # Initial UI feedback
    if not text and not audio_file:
        yield "❌ Error", "", "", "Please provide input."
        return

    # 1. Start Status
    yield "⏳ Transcribing/Processing...", "...", "...", "Starting pipeline..."

    transcribed_text = text

    # Handle audio transcription
    if audio_file:
        try:
            with open(audio_file, "rb") as f:
                files = {"audio": f}
                resp = requests.post(f"{BACKEND_URL}/transcribe", files=files, timeout=1000)

            if resp.status_code == 200:
                transcribed_text = resp.json().get("transcribed_text", text)
                yield "✅ Transcribed", "...", "...", f"Prompt: {transcribed_text[:50]}..."
            else:
                yield "❌ STT Failed", "", "", f"Error: {resp.text}"
                return
        except Exception as e:
            yield "❌ Connection Error", "", "", str(e)
            return

    # 2. Call backend for the Multi-Agent heavy lifting
    yield "🤖 Agents are working (BA -> QA -> SDET)...", "...", "...", "This usually takes 30-60 seconds."

    try:
        payload = {"requirements": transcribed_text, "use_rag": use_rag}
        # We use a long timeout because Llama 3 is generating 3 different artifacts
        resp = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=1000)

        if resp.status_code == 200:
            data = resp.json()

            # Format Compliance Result
            comp = data.get("compliance_evaluation", "")
            status_header = "✅ COMPLIANT" if "✅" in comp else "❌ REJECTED"

            yield (
                data.get("user_stories", ""),
                data.get("test_cases", ""),
                data.get("test_code", ""),
                f"### {status_header}\n\n{comp}"
            )
        else:
            yield "❌ Backend Error", "", "", f"Status {resp.status_code}: {resp.text}"

    except requests.exceptions.Timeout:
        yield "⏰ Timeout", "", "", "The backend is taking too long. Check if Ollama is running."
    except Exception as e:
        yield "❌ Error", "", "", str(e)


# ====================== GRADIO UI ======================
with gr.Blocks(title="AI SDET Pipeline", theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# 🚀 AI Requirement-to-Test Pipeline")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Business Requirements", lines=6, placeholder="Describe the feature...")
            audio_input = gr.Audio(label="Voice Input", sources=["microphone"], type="filepath")
            use_rag = gr.Checkbox(label="Use Knowledge Base (RAG)", value=True)

            generate_btn = gr.Button("🚀 Generate & Evaluate", variant="primary")

        with gr.Column(scale=2):
            # Status Indicator (Crucial for UX)
            status_box = gr.Markdown("### Status: Ready 🟢")

            with gr.Tabs():
                with gr.Tab("📋 User Stories"):
                    stories_out = gr.Markdown()
                with gr.Tab("✅ Test Cases"):
                    tests_out = gr.Markdown()
                with gr.Tab("💻 Selenium Code"):
                    code_out = gr.Code(language="python")
                with gr.Tab("📊 Compliance"):
                    compliance_out = gr.Markdown()

    # Link the button to the function
    # 'yield' in the function allows us to update the status_box in real-time
    generate_btn.click(
        fn=process_requirements,
        inputs=[text_input, audio_input, use_rag],
        outputs=[stories_out, tests_out, code_out, compliance_out],
        show_progress="full"
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)