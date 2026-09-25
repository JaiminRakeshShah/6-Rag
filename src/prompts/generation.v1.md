Answer the question using only the retrieved chunks. If they do not contain the answer, say the documents do not say, and do not invent a figure or a name.
Name every section you used. Copy section_id and section_name exactly as they are written on the chunk. Those are the same fields a gold supporting section records.
Reply with one JSON object and no other text:
{"answer": "...", "sections": [{"section_id": "...", "section_name": "..."}], "confidence": 0.0}
confidence is a number from 0 to 1.

Question: {question}

Chunks:
{chunks}
