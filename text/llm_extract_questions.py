import os
from pathlib import Path
import json
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configure these paths for your environment ─────────────────────────────────
TRANSCRIPTS_DIR = Path("CONFIGURE_ME/transcripts")   # directory of .txt transcript files
OUTPUT_DIR      = Path("CONFIGURE_ME/question_answers")  # JSON answers will be written here
MODEL_PATH      = Path("CONFIGURE_ME/models/llama")   # local Llama model directory
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Role: You are an AI clinical documentation assistant.\n\n"
    "Task: Answer a fixed set of questions based only on the provided interview transcript.\n\n"
    "Instructions:\n"
    "1.  Strict Structure: Simply list the answers for each question based on what is explicitly stated in the transcript. Do not provide any explanations, context, or additional text.\n"
    "2.  Strict Format: Each answer must be exactly one of the following options: 'YES', 'NO', 'To Some Extent', or 'Not Mentioned'.\n"
    "3.  Source Adherence: Every piece of information in your answers must come directly from the provided text.\n"
    "4.  Output Language: Every answer must be in English, regardless of the language of the input transcript."
)

PROMPT_TEXT = (
    "Can you answer these questions from this text which is an interview with a person only with 'YES' or 'NO' or 'To Some Extent'? If the question or corresponding answer is not found, answer 'Not Mentioned'.\n\n"
    "1. Have you felt emotionally and physically well lately?\n"
    "2. Have you noticed significant changes in your mood, such as feeling persistently sad, empty, or hopeless?\n"
    "3. Have you experienced difficulties with your sleep, such as trouble falling asleep, staying asleep, or waking up too early?\n"
    "4. Are you finding it challenging to concentrate on tasks or make decisions?\n"
    "5. Have you lost interest or pleasure in activities you used to enjoy?\n"
    "6. Have you ever been diagnosed with depression or experienced prolonged periods of feeling down or hopeless in the past?\n"
    "7. Have you ever been diagnosed with PTSD (Post-Traumatic Stress Disorder)?\n"
    "8. Have you been experiencing any financial problems recently?\n"
    "9. Do you find it challenging to socialize and prefer solitary activities, indicating introverted tendencies?\n"
    "10. Have you had thoughts of death or suicide, or have you made any suicide attempts?\n"
    "11. Have you ever served in the military?\n"
    "Always answer completely in English, regardless of the language of the input transcript."
)

MAX_CONTEXT_TOKENS = 40000
MAX_NEW_TOKENS = 500
TOKEN_SAFETY_MARGIN = 50
EXCLUDED_IDS = {177, 207, 299}


def get_file_id(filename: str) -> int | None:
    """Extracts the leading number from a filename for sorting or filtering."""
    try:
        return int(filename.split("_")[0])
    except (ValueError, IndexError):
        return None

def generate_answers(model, tokenizer, transcript: str, prompt_len: int, filename: str):
    """Generates answers for a single transcript using the loaded model."""
    max_transcript_tokens = (
        MAX_CONTEXT_TOKENS - prompt_len - MAX_NEW_TOKENS - TOKEN_SAFETY_MARGIN
    )

    transcript_ids = tokenizer.encode(transcript)
    if len(transcript_ids) > max_transcript_tokens:
        transcript_ids = transcript_ids[:max_transcript_tokens]

    transcript_trimmed = tokenizer.decode(transcript_ids, skip_special_tokens=True)

    user_message = (
        f"{PROMPT_TEXT}\n\n"
        f"--- BEGIN TRANSCRIPT ---\n{transcript_trimmed}\n--- END TRANSCRIPT ---\n\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    actual_input_tokens = input_ids.shape[-1]
    logging.info(f"[{filename}] Input tokens: {actual_input_tokens}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = outputs[0][actual_input_tokens:]
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    actual_output_tokens = len(output_ids)

    del input_ids, outputs, transcript_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return generated_text, actual_input_tokens, actual_output_tokens


def main():
    """The main function that runs the entire question-answering pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    logging.info("Model and tokenizer loaded successfully.")

    try:
        all_files = sorted(os.listdir(TRANSCRIPTS_DIR), key=get_file_id)
    except FileNotFoundError:
        logging.error(f"Transcript directory not found: {TRANSCRIPTS_DIR}")
        return

    files_to_process = []
    for filename in all_files:
        if not filename.endswith(".txt"):
            continue

        file_id = get_file_id(filename)
        if file_id is not None and file_id in EXCLUDED_IDS:
            logging.info(f"Skipping excluded file: {filename}")
            continue

        output_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".json"))
        if os.path.exists(output_path):
            continue

        files_to_process.append(filename)

    if not files_to_process:
        logging.info("No new transcripts to process. Exiting.")
        return
    logging.info(f"Found {len(files_to_process)} new transcripts to process.")

    messages_for_len_calc = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{PROMPT_TEXT}\n\n--- BEGIN TRANSCRIPT ---\n\n--- END TRANSCRIPT ---\n\n"},
    ]
    prompt_len = len(tokenizer.apply_chat_template(messages_for_len_calc, add_generation_prompt=True))

    for filename in tqdm(files_to_process, desc="Processing transcripts"):
        try:
            transcript_path = os.path.join(TRANSCRIPTS_DIR, filename)
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()

            answers, in_tokens, out_tokens = generate_answers(
                model, tokenizer, transcript_text, prompt_len, filename
            )

            output_data = {
                "question_answers": answers,
                "configured_max_input_tokens": MAX_CONTEXT_TOKENS,
                "configured_max_output_tokens": MAX_NEW_TOKENS,
                "actual_input_tokens": in_tokens,
                "actual_output_tokens": out_tokens,
            }

            output_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}", exc_info=True)

    logging.info(f"Processing complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
