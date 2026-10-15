import pypdf
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    reader = pypdf.PdfReader('D:/SARVAM/sarvam_mini_ps.pdf')
    print(f"Total Pages: {len(reader.pages)}")
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        first_few = " | ".join(lines[:3]) if lines else "EMPTY"
        last_few = " | ".join(lines[-3:]) if lines else "EMPTY"
        print(f"\n--- PAGE {idx+1} ({len(lines)} lines) ---")
        print(f"  START: {first_few[:120]}")
        print(f"  END:   {last_few[:120]}")

if __name__ == "__main__":
    main()
