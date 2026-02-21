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
PROMPTS_PATH = ""
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

ANSWER_MAP_BY_TYPE = {
    # YES / TRUE
    "yes": 1, "Yes": 1, "YES": 1, "yeah": 1, "y": 1, "true": 1, "True": 1,

    # NO / FALSE
    "no": 0, "No": 0, "NO": 0, "nah": 0, "n": 0, "false": 0, "False": 0,

    # SCALE
    "0": 0, "zero": 0,
    "1": 1, "one": 1,
    "2": 2, "two": 2,
    "3": 3, "three": 3, 
    "4": 4, "four": 4,
    "5": 5, "five": 5,

    # CLASSIFICATION (HOMO / TRANS / NEUTRAL)
    "A": 0, "a": 0, "homophobic": 0, "Homophobic": 0, "gay people": 0,
    "B": 1, "b": 1, "transphobic": 1, "Transphobic": 1, "transgender people": 1,
    "C": 2, "c": 2, "neither": 2, "Neither": 2, "neutral": 2, "none": 2, "no group": 2,

    # TARGET TYPE
    "sexual orientation": 0, "orientation": 0,
    "gender identity": 1, "gender": 1,
    "neither": 2, "neutral": 2, "none": 2, "no target": 2,

    "INV": 6,
}

answer_keys = list(ANSWER_MAP_BY_TYPE.keys())

def find_answer(text, keys):
    for key in keys:
        if key in text:
            return key
    return ""


sys_prompt = """You are a labeling system.

You will be shown a meme image and then asked a single question about it.

Rules:
- Do NOT explain your reasoning.
- Do NOT add extra words, punctuation, or commentary.
- Output only the answer.
- Be concise and deterministic.
- If unsure, choose the closest valid answer.
- The meme may contain English, Hindi, or Chinese text.
- Focus on meaning rather than language.
- Watch for sarcasm or parody.
- If the meme criticizes hateful views, do NOT mark it as hateful.
- Never include explanations.
- Never include multiple answers."""

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

df = pd.read_json(PROMPTS_PATH, lines=True)

image_paths = []

for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.gif", "*.GIF", "*.webp", "*.WEBP"):
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

image_paths = sorted(
    image_paths,
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)

print(len(df), "prompts loaded.")
print(len(image_paths), "images found.")

all_results = []

for image_path in tqdm(image_paths, total=len(image_paths)):
    filename = os.path.basename(image_path)       
    file_id = int(os.path.splitext(filename)[0])   
    image = load_image_safe(image_path)
    answers = [file_id]

    for _, row in df.iterrows():
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
                    {"type": "text", "text": row.question},
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
            out = model.generate(**inputs, max_new_tokens=5)

            new_tokens = out[:, in_len:]
            decoded = processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            generated_text = find_answer(decoded, row.acceptable_answers)
            if generated_text:
                answers.append(ANSWER_MAP_BY_TYPE[generated_text])
                break

        if not generated_text:
            answers.append(ANSWER_MAP_BY_TYPE["INV"])

    all_results.append(answers)

    new_df = pd.DataFrame(all_results)
    new_df.columns = ["id"] + [f"feat_{i}" for i in range(new_df.shape[1])]
    new_df.to_csv(OUTPUT_CSV, index=False)