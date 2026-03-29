import pypdf
import sys

def extract():
    with open('lungnodulemamba.pdf.pdf', 'rb') as f:
        reader = pypdf.PdfReader(f)
        text = ''
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + '\n'
                
    with open('pdf_text.txt', 'w', encoding='utf-8') as fout:
        fout.write(text)

if __name__ == '__main__':
    extract()
