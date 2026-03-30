import csv
from math import e
import re
import ast
import time
from scipy.special import j0
from difflib import SequenceMatcher


parsed_data=[]

with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        parsed_row = []
        for cell in row:
            try:
                # Convert the string representation of a list to an actual list
                parsed_cell = ast.literal_eval(cell.strip())
            except (SyntaxError, ValueError):
                # Fallback to raw string if parsing fails
                parsed_cell = cell.strip()
            parsed_row.append(parsed_cell)
        if parsed_row not in parsed_data:
            parsed_data.append(parsed_row)


for i in range(1, (len(parsed_data)-1)):
    parsed_dataout=[]
    for j in range(2,9):
        parsed_dataout.append(parsed_data[i][j])
    parsed_dataout.append(parsed_data[i][109])
    parsed_dataout.append(parsed_data[i][162])
    parsed_dataout.append(parsed_data[i][163])
    parsed_dataout.append(parsed_data[i][198])
    for j in range(211,387):
        parsed_dataout.append(parsed_data[i][j])

    with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile2:
        csvwriter = csv.writer(csvfile2)
        csvwriter.writerow(parsed_dataout)






