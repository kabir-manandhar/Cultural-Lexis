import fitz  # PyMuPDF
import re


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text



def parse_questions(text):
    pattern = re.compile(
        r"Q(?P<id>\d+): (?P<question>.*?)\n"
        r"a\) (?P<option_a>.*?)\n"
        r"b\) (?P<option_b>.*?)\n"
        r"c\) (?P<option_c>.*?)\n"
        r"d\) (?P<option_d>.*?)\n",
        re.DOTALL
    )
    questions = []
    for match in pattern.finditer(text):
        questions.append({
            "Question ID": match.group("id"),
            "Question": match.group("question").strip(),
            "Option A": match.group("option_a").strip(),
            "Option B": match.group("option_b").strip(),
            "Option C": match.group("option_c").strip(),
            "Option D": match.group("option_d").strip(),
        })
    return questions

pdf_path = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/WV_Bench/English_Questions.pdf"

text = extract_text_from_pdf(pdf_path)

import re

questions_pattern = r"(Q\d{1,3})\s+(.*?)\n((?:\d+\s+.+?\n)+)"
matches = re.findall(questions_pattern, text, re.DOTALL)
questions=[]
for match in matches:
    question_id, question_text, options = match
    questions.append((question_id, question_text, options))
    print(f"Question ID: {question_id}\nQuestion: {question_text.strip()}\nOptions:\n{options.strip()}\n")

breakpoint()
questions = parse_questions(text)