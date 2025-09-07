import os
from openpyxl import Workbook

question_inputs = input("Введите список вопросов:\n")
answer_inputs = input("Введите список ответов на вопросы:\n")

questions = [q.strip() for q in question_inputs.split(";") if q.strip()]
answers = [a.strip() for a in answer_inputs.split(";") if a.strip()]

if (len(questions) != len(answers)):
    print("Questions & answers quality does not matching")
    exit(1)
    
wb = Workbook()
ws = wb.active
ws.title = "Q&A's"

ws.append(["Вопрос", "Ответ"])

for q, a in zip(questions, answers):
    ws.append([q, a])

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "exels")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
file_path = os.path.join(folder_path, "Q&A_Part.xlsx")
wb.save(file_path)

print("The file has been successfully saved")