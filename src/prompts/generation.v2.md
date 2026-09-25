Answer the question using only the retrieved chunks.
If the question asks for the current document, use status current. If it asks for the superseded or archived document, use status superseded. When both copies state the same fact, cite the copy the question asked for.
If two current sections disagree, state both. Do not drop one and do not decline.
If the chunks do not contain the asked fact, answer only "The documents do not say." Leave sections empty, set confidence to 0, and do not add a nearby figure or name.
A yes or no question starts with Yes or No.
Cite section_id and section_name exactly as written on the chunk. Do not put a chunk_id in section_id.
Reply with one JSON object and no other text:
{"answer": "...", "sections": [{"section_id": "...", "section_name": "..."}], "confidence": 0.0}
confidence is a number from 0 to 1.

Example 1. The question asks for the current copy.
Question: What paper-use cut does the current Office Note set for 2026?
Chunks:
chunk_id: Office_Note_2019#3:0
section_id: Office_Note_2019#3
section_name: Targets
filename: Office_Note_2019.md
status: superseded
body:
Cut office paper use 15% by 2026.

chunk_id: Office_Note_2024#3:0
section_id: Office_Note_2024#3
section_name: Targets
filename: Office_Note_2024.md
status: current
body:
Cut office paper use 40% by 2026.

{"answer": "40% by 2026.", "sections": [{"section_id": "Office_Note_2024#3", "section_name": "Targets"}], "confidence": 0.9}

Example 2. The asked fact is absent. A nearby name stays out of the answer.
Question: What is Priya Shah's mobile number?
Chunks:
chunk_id: Travel_Note#1:0
section_id: Travel_Note#1
section_name: Sign Off
filename: Travel_Note.md
status: current
body:
Signed by Priya Shah, Finance Director. Desk extension 4400.

{"answer": "The documents do not say.", "sections": [], "confidence": 0.0}

Example 3. Two current sections disagree, so the answer states both.
Question: How often does the current Travel Note say managers file expense reports?
Chunks:
chunk_id: Travel_Note#4:0
section_id: Travel_Note#4
section_name: Managers
filename: Travel_Note.md
status: current
body:
Managers file a monthly expense report.

chunk_id: Travel_Note#5:0
section_id: Travel_Note#5
section_name: Reporting
filename: Travel_Note.md
status: current
body:
Managers file an expense report each quarter.

{"answer": "The Managers section says monthly. The Reporting section says each quarter.", "sections": [{"section_id": "Travel_Note#4", "section_name": "Managers"}, {"section_id": "Travel_Note#5", "section_name": "Reporting"}], "confidence": 0.7}

Example 4. A yes or no question starts with Yes or No.
Question: Does the current Travel Note apply only to headquarters?
Chunks:
chunk_id: Travel_Note#2:0
section_id: Travel_Note#2
section_name: Scope
filename: Travel_Note.md
status: current
body:
This note applies to headquarters and field offices.

{"answer": "No. It applies to headquarters and field offices.", "sections": [{"section_id": "Travel_Note#2", "section_name": "Scope"}], "confidence": 0.9}

Question: {question}

Chunks:
{chunks}
