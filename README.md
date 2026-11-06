# Driver Drowsiness Detection + Gemini RAG

## File
- `drowsiness_gui_v3_rag.py`: ban app nhan dien goc co them nut Tro ly an toan (RAG).
- `rag_gemini.py`: doc PDF/DOCX/TXT/MD, chia chunk, luu ChromaDB va hoi Gemini.
- `drowsiness_gui_v3.py`: ban goc duoc giu lai de du phong.
- `documents/`: dat tai lieu an toan cua doanh nghiep vao day.
- `chroma_db/`: tu tao khi chay, khong can tao thu cong.

## Cai dat trong venv
```powershell
python -m pip install -r requirements-rag.txt
```
App camera goc van can cac thu vien da dung truoc do (opencv, mediapipe, pillow, numpy...).

## Dat API key trong PowerShell
```powershell
$env:GEMINI_API_KEY="API_KEY_CUA_BAN"
```
Key chi ton tai trong cua so PowerShell hien tai. Khong chia se key va khong commit key len GitHub.

## Chay app
```powershell
python drowsiness_gui_v3_rag.py
```
Mo ung dung, vao man hinh chinh va bam `Tro ly an toan (RAG)`. Dat tai lieu trong `documents/`, bam `Nap tai lieu trong documents`, roi hoi dap.

## Chay RAG rieng de thu
```powershell
python rag_gemini.py
```

## Cach RAG hoat dong
1. Trich xuat text tu PDF/DOCX/TXT/MD.
2. Chia van ban thanh chunks ~1200 ky tu, overlap ~180 ky tu.
3. Goi `gemini-embedding-001` cho tung chunk va luu vector vao ChromaDB.
4. Embedding cau hoi, truy xuat 5 chunk gan nhat.
5. Gui cac chunk va cau hoi toi `gemini-3.5-flash-lite` de sinh cau tra loi, kem danh sach nguon.

Luu y:
- PDF scan anh can OCR truoc.
- Moi lan bam nap, chunk ID duoc tinh tu noi dung; chunk khong doi se duoc upsert lai. Neu tai lieu da bi xoa/doi, chunk cu khong tu dong bi xoa khoi DB; neu can lam sach kho, dung chuong trinh khi da sao luu va xoa thu muc `chroma_db/` de tao lai.
- Tai lieu noi bo co the chua thong tin nhay cam; chi nap tai lieu duoc phep gui toi dich vu Gemini.
- Tro ly la cong cu tham khao, khong thay the canh bao thoi gian thuc, nguong CV hay quy trinh chinh thuc.
- Loi 503 la loi dich vu tam thoi; neu xay ra, thu lai sau va kiem tra tinh kha dung cua model.
