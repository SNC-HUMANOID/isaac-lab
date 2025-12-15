#!/bin/bash
# สคริปต์สำหรับแปลง Markdown เป็น PDF พร้อม syntax highlighting

echo "=== Isaac Lab Markdown to PDF Converter ==="
echo ""

# ตรวจสอบว่าติดตั้ง pandoc และ wkhtmltopdf หรือยัง
if ! command -v pandoc &> /dev/null; then
    echo "❌ ไม่พบ pandoc - กรุณาติดตั้ง:"
    echo "   Ubuntu/Debian: sudo apt-get install pandoc texlive-xetex texlive-fonts-recommended"
    echo "   หรือ: sudo apt-get install pandoc wkhtmltopdf"
    exit 1
fi

INPUT_FILE="คู่มือสร้างโมเดลหุ่นยนต์_Isaac_Lab.md"
OUTPUT_FILE="คู่มือสร้างโมเดลหุ่นยนต์_Isaac_Lab.pdf"

echo "📄 Input: $INPUT_FILE"
echo "📄 Output: $OUTPUT_FILE"
echo ""

# วิธีที่ 1: ใช้ pandoc + LaTeX (คุณภาพดีที่สุด)
if command -v xelatex &> /dev/null; then
    echo "✅ ใช้ pandoc + XeLaTeX (คุณภาพสูง)"
    pandoc "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        --pdf-engine=xelatex \
        --highlight-style=tango \
        --toc \
        -V geometry:margin=1in \
        -V linkcolor:blue \
        -V fontsize=11pt \
        -V mainfont="TH Sarabun New" \
        --metadata title="คู่มือสร้างโมเดลหุ่นยนต์ Isaac Lab"

# วิธีที่ 2: ใช้ wkhtmltopdf (ง่ายกว่า)
elif command -v wkhtmltopdf &> /dev/null; then
    echo "✅ ใช้ pandoc + wkhtmltopdf"
    pandoc "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        -t html5 \
        --highlight-style=tango \
        --toc \
        --self-contained \
        --pdf-engine=wkhtmltopdf

# วิธีที่ 3: แปลงเป็น HTML แทน
else
    echo "⚠️  ไม่พบ PDF engine - แปลงเป็น HTML แทน"
    OUTPUT_FILE="คู่มือสร้างโมเดลหุ่นยนต์_Isaac_Lab.html"
    pandoc "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        --standalone \
        --highlight-style=tango \
        --toc \
        --self-contained \
        -c https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css

    echo ""
    echo "💡 เปิดไฟล์ HTML ด้วย browser แล้วกด Ctrl+P เพื่อ Save as PDF"
fi

echo ""
echo "✅ เสร็จสิ้น: $OUTPUT_FILE"
