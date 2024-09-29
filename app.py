import gradio as gr
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

def model_inference(images):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    images = [{"type": "image", "image": Image.open(image[0])} for image in images]
    
    messages = [{"role": "user", "content": images}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    model = model.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    del model
    del processor
    return output_text[0]

def search_and_highlight(text, keywords):
    if not keywords:
        return text

    keywords = [kw.strip().lower() for kw in keywords.split(',')]
    highlighted_text = text

    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(f'**{keyword}**', highlighted_text)

    return highlighted_text

def extract_and_search(images, keywords):
    extracted_text = model_inference(images)
    highlighted_text = search_and_highlight(extracted_text, keywords)
    return extracted_text, highlighted_text

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        output_gallery = gr.Gallery(label="Image", height=300, show_label=True)
        keywords = gr.Textbox(placeholder="Enter keywords to search (comma-separated)", label="Search Keywords")
    
    extract_button = gr.Button("Extract Text and Search", variant="primary")
    
    with gr.Row():
        raw_output = gr.Textbox(label="Interpreted Text")
        highlighted_output = gr.Markdown(label="Highlighted Search Results")

    extract_button.click(extract_and_search, inputs=[output_gallery, keywords], outputs=[raw_output, highlighted_output])

if __name__ == "__main__":
    demo.queue(max_size=10).launch(share=True)