import glob
import numpy as np
import os
import pandas as pd
import random
import torch

from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoProcessor
from transformers import Qwen3VLMoeForConditionalGeneration

IMAGE_DIR = ""
OUTPUT_CSV = ""
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID = ["Homophobia", "Transphobia", "Non_Anti_LGBT", 
         "Homophobia.", "Transphobia.", "Non_Anti_LGBT.", 
         "homophobia", "transphobia", "non_anti_lgbt", 
         "homophobia.", "transphobia.", "non_anti_lgbt."]

sys_prompt = """You are a labeling system.

You will be shown a meme image.

Your task is to classify the meme into exactly one of the following categories:
- Homophobia
- Transphobia
- Non_Anti_LGBT

Rules:
- Do NOT explain your reasoning.
- Do NOT add extra words, punctuation, or commentary.
- Output only one of the three labels exactly as written.
- Be concise and deterministic.
- If unsure, choose the closest valid label.
- The meme may contain English, Hindi, or Chinese text.
- Focus on meaning rather than language.
- Watch for sarcasm, irony, or parody.
- If the meme criticizes or mocks homophobia or transphobia, classify it as Non_Anti_LGBT.
- The label should reflect the target and intent of the meme, not just keywords.
- Never include explanations.
- Never include multiple labels."""


def load_image_safe(path):
    img = Image.open(path)

    # If it's a GIF (or animated), grab first frame
    if getattr(img, "is_animated", False):
        img.seek(0)

    return img.convert("RGB")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    attn_implementation="sdpa",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

image_paths = []

for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.gif", "*.GIF", "*.webp", "*.WEBP"):
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

image_paths = sorted(
    image_paths,
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)

print(len(image_paths), "images found.")

all_results = []

for image_path in tqdm(image_paths, total=len(image_paths)):
    filename = os.path.basename(image_path)       
    file_id = int(os.path.splitext(filename)[0])   
    image = load_image_safe(image_path)
    answers = [file_id]

    generated_text = None

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)
    in_len = inputs["input_ids"].shape[1]

    for _ in range(10):
        out = model.generate(**inputs, max_new_tokens=20)

        new_tokens = out[:, in_len:]
        decoded = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if decoded in VALID:
            break

    answers.append(decoded)
    all_results.append(answers)

    new_df = pd.DataFrame(all_results)
    new_df.columns = ["id", "label"]
    new_df.to_csv(OUTPUT_CSV, index=False)