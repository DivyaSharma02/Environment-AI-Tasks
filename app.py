import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import networkx as nx
import matplotlib.pyplot as plt
import tempfile

# Load pipelines
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")  # Replace with env-trained model
ner_pipe = pipeline("ner", grouped_entities=True)
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Image generation (Stable Diffusion)
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
sd_pipe = sd_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 1. Sentence Classification
def classify_text(text):
    result = classifier(text)
    return result[0]['label']

# 2. Image Generation
def generate_image(prompt):
    image = sd_pipe(prompt).images[0]
    return image

# 3. NER and Graph
def ner_graph(text):
    entities = ner_pipe(text)
    G = nx.Graph()
    for ent in entities:
        G.add_node(ent['word'], label=ent['entity_group'])
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            G.add_edge(entities[i]['word'], entities[j]['word'])

    # Draw and return image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000)
        plt.savefig(tmp.name)
        plt.close()
        return tmp.name

# 4. Fill in the blank
def fill_masked(text):
    results = fill_mask(text)
    return [r['sequence'] for r in results]

# Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 Environment AI Tasks using Hugging Face 🤗")
    
    with gr.Tab("1️⃣ Sentence Classification"):
        inp1 = gr.Textbox(label="Enter environmental sentence")
        out1 = gr.Textbox(label="Predicted Category")
        btn1 = gr.Button("Classify")
        btn1.click(classify_text, inp1, out1)

    with gr.Tab("2️⃣ Image Generation from Prompt"):
        inp2 = gr.Textbox(label="Describe an environment scene")
        out2 = gr.Image(label="Generated Image")
        btn2 = gr.Button("Generate Image")
        btn2.click(generate_image, inp2, out2)

    with gr.Tab("3️⃣ Named Entity Recognition with Graph"):
        inp3 = gr.Textbox(label="Enter a sentence about environment")
        out3 = gr.Image(label="NER Graph")
        btn3 = gr.Button("Generate NER Graph")
        btn3.click(ner_graph, inp3, out3)

    with gr.Tab("4️⃣ Fill in the Blank (Masked Word)"):
        inp4 = gr.Textbox(label="Enter sentence with [MASK] token")
        out4 = gr.JSON(label="Predicted Sentences")
        btn4 = gr.Button("Predict Mask")
        btn4.click(fill_masked, inp4, out4)

demo.launch()
