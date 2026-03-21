import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PAGE_RE = re.compile(
	r"^(?P<doc_id>(?P<doc_type>constituency|party_list)_(?P<province_code>\d+)_(?P<constituency>\d+))(?:_page(?P<page>\d+))?\.png$",
	re.IGNORECASE,
)

THAI_DIGIT_MAP = str.maketrans({
	"๐": "0",
	"๑": "1",
	"๒": "2",
	"๓": "3",
	"๔": "4",
	"๕": "5",
	"๖": "6",
	"๗": "7",
	"๘": "8",
	"๙": "9",
})


def normalize_votes(value: object) -> str:
	if value is None:
		return "0"
	text = str(value).translate(THAI_DIGIT_MAP)
	text = text.replace(",", "")
	digits = "".join(ch for ch in text if ch.isdigit())
	return digits if digits else "0"


def extract_json_object(text: str) -> dict:
	start = text.find("{")
	end = text.rfind("}")
	if start < 0 or end < 0 or end <= start:
		return {}
	candidate = text[start : end + 1]
	try:
		return json.loads(candidate)
	except json.JSONDecodeError:
		return {}


def parse_filename(path: Path) -> tuple[str, int] | None:
	match = PAGE_RE.match(path.name)
	if not match:
		return None
	doc_id = match.group("doc_id")
	page = int(match.group("page")) if match.group("page") else 1
	return doc_id, page


def group_document_pages(image_dir: Path) -> dict[str, list[Path]]:
	grouped: dict[str, list[tuple[int, Path]]] = {}
	for path in image_dir.glob("*.png"):
		parsed = parse_filename(path)
		if parsed is None:
			continue
		doc_id, page = parsed
		grouped.setdefault(doc_id, []).append((page, path))

	result: dict[str, list[Path]] = {}
	for doc_id, items in grouped.items():
		items.sort(key=lambda x: x[0])
		result[doc_id] = [path for _, path in items]
	return result


def detect_images_per_call() -> int:
	if not torch.cuda.is_available():
		return 1
	total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
	if total_gb >= 44:
		return 6
	if total_gb >= 30:
		return 4
	if total_gb >= 22:
		return 3
	if total_gb >= 16:
		return 2
	return 1


def clear_gpu_memory() -> None:
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()


def is_oom_error(exc: RuntimeError) -> bool:
	msg = str(exc).lower()
	return "out of memory" in msg and "cuda" in msg


def build_messages(row_meta: list[tuple[int, str]]):
	row_lines = "\n".join([f"- row_num={row_num} | party_name={party}" for row_num, party in row_meta])
	prompt = (
		"You are extracting Thai election vote counts from Form สส.6/1 images.\n"
		"Task: return ONLY JSON with visible rows and their vote counts.\n"
		"Rules:\n"
		"1) Output strict JSON object with key 'rows'.\n"
		"2) rows is a list of objects: {'row_num': int, 'votes': 'digits'}.\n"
		"3) Include ONLY rows clearly visible in the provided images.\n"
		"4) Convert Thai digits to Arabic digits 0-9.\n"
		"5) votes must contain digits only (no commas, spaces, text).\n"
		"6) If a row is visible but unreadable, set votes='0'.\n"
		"7) Do not output markdown, explanations, or extra keys.\n"
		"Candidate rows for this document:\n"
		f"{row_lines}\n"
		"Expected format example:\n"
		"{\"rows\": [{\"row_num\": 1, \"votes\": \"123\"}, {\"row_num\": 2, \"votes\": \"0\"}]}"
	)
	return [
		{
			"role": "user",
			"content": [
				{"type": "text", "text": prompt},
				{"type": "image"},
			],
		}
	]


@dataclass
class OCRState:
	device: torch.device
	model: Qwen2_5_VLForConditionalGeneration
	processor: AutoProcessor
	max_new_tokens: int


def load_model(model_name: str, processor_name: str) -> OCRState:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
	kwargs = {"torch_dtype": dtype}
	if device.type == "cuda":
		kwargs["attn_implementation"] = "flash_attention_2"

	try:
		model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
	except Exception:
		kwargs.pop("attn_implementation", None)
		model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)

	model = model.eval().to(device)
	processor = AutoProcessor.from_pretrained(processor_name)
	return OCRState(
		device=device,
		model=model,
		processor=processor,
		max_new_tokens=3072,
	)


def run_inference_once(state: OCRState, row_meta: list[tuple[int, str]], image_paths: list[Path]) -> dict[int, str]:
	messages = build_messages(row_meta)
	images = [Image.open(path).convert("RGB") for path in image_paths]
	try:
		text_inputs = [
			state.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
			for _ in images
		]
		inputs = state.processor(text=text_inputs, images=images, padding=True, return_tensors="pt")
		inputs = {k: v.to(state.device) for k, v in inputs.items()}

		with torch.inference_mode():
			output = state.model.generate(
				**inputs,
				do_sample=False,
				temperature=0.0,
				max_new_tokens=state.max_new_tokens,
			)

		prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
		new_tokens = [output[i, p:] for i, p in enumerate(prompt_lengths)]
		decoded = state.processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

		merged = {}
		for text in decoded:
			parsed = extract_json_object(text)
			for item in parsed.get("rows", []):
				try:
					row_num = int(item.get("row_num"))
				except Exception:
					continue
				votes = normalize_votes(item.get("votes", "0"))
				if votes != "0":
					merged[row_num] = votes
				elif row_num not in merged:
					merged[row_num] = "0"
		return merged
	finally:
		for img in images:
			img.close()


def run_inference_chunk_safe(
	state: OCRState,
	row_meta: list[tuple[int, str]],
	image_paths: list[Path],
) -> dict[int, str]:
	if not image_paths:
		return {}
	try:
		return run_inference_once(state, row_meta, image_paths)
	except RuntimeError as exc:
		if state.device.type != "cuda" or not is_oom_error(exc):
			raise
		clear_gpu_memory()
		if len(image_paths) == 1:
			return {}
		half = math.ceil(len(image_paths) / 2)
		left = run_inference_chunk_safe(state, row_meta, image_paths[:half])
		right = run_inference_chunk_safe(state, row_meta, image_paths[half:])
		merged = left.copy()
		for k, v in right.items():
			if v != "0" or k not in merged:
				merged[k] = v
		return merged


def run_document(
	state: OCRState,
	pages: list[Path],
	row_meta: list[tuple[int, str]],
	images_per_call: int,
) -> dict[int, str]:
	votes_by_row: dict[int, str] = {}
	step = max(1, images_per_call)
	for idx in range(0, len(pages), step):
		chunk = pages[idx : idx + step]
		partial = run_inference_chunk_safe(state, row_meta, chunk)
		for row_num, votes in partial.items():
			if votes != "0" or row_num not in votes_by_row:
				votes_by_row[row_num] = votes

	for row_num, _ in row_meta:
		votes_by_row.setdefault(row_num, "0")
	return votes_by_row


def build_submission(
	template_df: pd.DataFrame,
	grouped_pages: dict[str, list[Path]],
	state: OCRState,
	images_per_call: int,
	limit_docs: int | None,
) -> tuple[pd.DataFrame, dict[str, dict[int, str]]]:
	doc_ids = template_df["doc_id"].drop_duplicates().tolist()
	if limit_docs is not None:
		doc_ids = doc_ids[:limit_docs]

	votes_by_doc: dict[str, dict[int, str]] = {}
	for doc_id in tqdm(doc_ids, desc="Documents"):
		doc_rows = template_df[template_df["doc_id"] == doc_id].copy()
		doc_rows = doc_rows.sort_values("row_num")
		row_meta = list(zip(doc_rows["row_num"].astype(int), doc_rows["party_name"].astype(str)))
		pages = grouped_pages.get(doc_id, [])
		if not pages:
			votes_by_doc[doc_id] = {row_num: "0" for row_num, _ in row_meta}
			continue
		votes_by_doc[doc_id] = run_document(state, pages, row_meta, images_per_call)

	output = template_df.copy()

	def lookup_votes(row: pd.Series) -> str:
		doc_map = votes_by_doc.get(row["doc_id"], {})
		return normalize_votes(doc_map.get(int(row["row_num"]), "0"))

	output["votes"] = output.apply(lookup_votes, axis=1)
	output = output[["id", "votes"]]
	return output, votes_by_doc


def main() -> None:
	parser = argparse.ArgumentParser(description="OCR vote extraction pipeline for Super AI SS6")
	parser.add_argument("--template_csv", type=str, required=True)
	parser.add_argument("--image_dir", type=str, required=True)
	parser.add_argument("--output_csv", type=str, default="submission.csv")
	parser.add_argument("--debug_json", type=str, default="votes_debug.json")
	parser.add_argument("--model_name", type=str, default="allenai/olmOCR-2-7B-1025")
	parser.add_argument("--processor_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
	parser.add_argument("--images_per_call", type=int, default=0)
	parser.add_argument("--limit_docs", type=int, default=0)
	args = parser.parse_args()

	template_path = Path(args.template_csv)
	image_dir = Path(args.image_dir)
	output_path = Path(args.output_csv)
	debug_path = Path(args.debug_json)

	template_df = pd.read_csv(template_path)
	required_cols = {"id", "doc_id", "row_num", "party_name"}
	missing = required_cols - set(template_df.columns)
	if missing:
		raise ValueError(f"Missing required columns in template: {missing}")

	template_df["row_num"] = template_df["row_num"].astype(int)
	template_df["id"] = template_df["id"].astype(str)
	template_df["doc_id"] = template_df["doc_id"].astype(str)
	template_df["party_name"] = template_df["party_name"].astype(str)

	grouped_pages = group_document_pages(image_dir)
	state = load_model(args.model_name, args.processor_name)

	if args.images_per_call > 0:
		images_per_call = args.images_per_call
	else:
		images_per_call = detect_images_per_call()

	limit_docs = args.limit_docs if args.limit_docs > 0 else None

	submission_df, debug_data = build_submission(
		template_df=template_df,
		grouped_pages=grouped_pages,
		state=state,
		images_per_call=images_per_call,
		limit_docs=limit_docs,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	submission_df.to_csv(output_path, index=False)
	with open(debug_path, "w", encoding="utf-8") as f:
		json.dump(debug_data, f, ensure_ascii=False, indent=2)

	print(f"Saved submission: {output_path}")
	print(f"Saved debug votes map: {debug_path}")
	print(f"Rows in submission: {len(submission_df)}")
	print(f"Images per call used: {images_per_call}")


if __name__ == "__main__":
	main()
