"""
Script to clean text and construct a structured dataset from Shariaa Standards.
"""
import re
import pandas as pd
from pathlib import Path

# Get project root directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

CLEAN_TEXT_DIR = PROJECT_ROOT / "clean_text"
DATASET_DIR = PROJECT_ROOT / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def clean_text_and_extract_standards(text_file):
    """Clean text and extract standards."""
    print(f"\nReading: {text_file}")
    text = text_file.read_text(encoding="utf-8")
    print(f"Original size: {len(text):,} characters, {len(text.splitlines()):,} lines")
    
    # Step 1: Find first standard and remove everything before it
    # Handle different apostrophe characters (regular ' or Unicode U+2019)
    first_standard_match = re.search(r"Shari[''\u2019]ah Standard No\.\s*\((\d+)\)", text, re.IGNORECASE)
    
    if not first_standard_match:
        print("ERROR: Could not find first standard!")
        return None
    
    start_pos = first_standard_match.start()
    print(f"\nFound first standard at position {start_pos} (line ~{text[:start_pos].count(chr(10)) + 1})")
    
    # Keep everything from first standard onwards
    cleaned_text = text[start_pos:]
    print(f"After removing pre-standard content: {len(cleaned_text):,} characters")
    
    # Step 2: Remove table of contents and metadata
    lines = cleaned_text.splitlines()
    final_lines = []
    in_toc = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Skip table of contents
        if line_stripped == "Contents" or line_stripped == "Table of Contents":
            in_toc = True
            continue
        
        if in_toc:
            # End TOC when we find a standard or section number
            if re.search(r"Shari[''\u2019]ah Standard No\.|^\d+//", line_stripped, re.IGNORECASE):
                in_toc = False
                final_lines.append(line)
            elif line_stripped == "Subject Page" or re.match(r'^[A-Z][^:]*\.{3,}\s*\d+$', line_stripped):
                # Skip TOC entry lines
                continue
            else:
                # End of TOC
                in_toc = False
                final_lines.append(line)
            continue
        
        # Skip "Subject Page" lines
        if line_stripped == "Subject Page":
            continue
        
        # Skip Preface with page numbers (e.g., "Preface ......................... 51")
        if re.match(r'^Preface\s*\.{3,}\s*\d+$', line_stripped):
            # Skip until Statement of the Standard or section number
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if re.search(r"Statement of the Standard|^\d+//", next_line, re.IGNORECASE):
                    break
                j += 1
            continue
        
        # Skip page number references (lines ending with dots and numbers)
        if re.match(r'^[A-Z][^:]*\.{3,}\s*\d+$', line_stripped):
            continue
        
        # Keep all other lines
        final_lines.append(line)
    
    cleaned_text = '\n'.join(final_lines)
    # Remove excessive blank lines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    print(f"After removing TOC and metadata: {len(cleaned_text):,} characters, {len(final_lines):,} lines")
    
    # Step 3: Extract only content that follows section numbers (e.g., "2//1", "2//1//1", etc.)
    lines = cleaned_text.splitlines()
    sections_data = []
    
    # Pattern to match section numbers: 2//1, 2//1//1, 3//5, etc.
    section_pattern = re.compile(r'^(\d+)(?://(\d+)(?://(\d+))?)?\s+(.+)$')
    
    # Pattern to detect table of contents entries
    # TOC entries typically have: dots (3+), page numbers at end, or "Appendix" with dots
    toc_pattern = re.compile(r'.*\.{3,}\s*\d+$|^Appendix\s*\([a-z]\):.*\.{3,}\s*\d+$', re.IGNORECASE)
    
    def is_toc_entry(line):
        """Check if a line is a table of contents entry."""
        line_stripped = line.strip()
        
        # Check for TOC patterns:
        # 1. Lines ending with multiple dots and page numbers
        if re.search(r'\.{3,}\s*\d+$', line_stripped):
            return True
        
        # 2. Lines with section numbers followed by dots and page numbers
        # e.g., "2//1,2//1,Default in payment ....................................................... 88,89,755"
        if re.match(r'^\d+//\d+(?://\d+)?,.*\.{3,}\s*\d+', line_stripped):
            return True
        
        # 3. Lines with numbered items followed by dots and page numbers
        # e.g., "1. Scope of the Standard .............................................................................. 88"
        if re.match(r'^\d+\.\s+.*\.{3,}\s*\d+$', line_stripped):
            return True
        
        # 4. Appendix entries with dots and page numbers
        if re.match(r'^Appendix\s*\([a-z]\):.*\.{3,}\s*\d+$', line_stripped, re.IGNORECASE):
            return True
        
        # 5. Lines that are just section numbers with commas and page numbers
        # e.g., "2//1,2//1,Default in payment by a debtor ....................................................... 88,89,755"
        if re.match(r'^\d+//\d+(?://\d+)?,\d+//\d+(?://\d+)?,.*\.{3,}', line_stripped):
            return True
        
        return False
    
    current_section = None
    current_content = []
    
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # Skip TOC entries
        if is_toc_entry(line_stripped):
            continue
        
        # Skip lines that are just "Appendices" or similar headers
        if line_stripped in ["Appendices", "Appendix"]:
            continue
        
        # Skip lines that are just section numbers with commas (TOC format)
        # e.g., "2//1,2//1," followed by content with dots
        if re.match(r'^\d+//\d+(?://\d+)?,\d+//\d+(?://\d+)?,', line_stripped):
            # Check if it ends with dots and numbers (TOC entry)
            if re.search(r'\.{3,}\s*\d+', line_stripped):
                continue
        
        # Check if this line starts with a section number
        match = section_pattern.match(line_stripped)
        
        if match:
            # Save previous section if exists
            if current_section is not None and current_content:
                section_text = '\n'.join(current_content).strip()
                if section_text:
                    sections_data.append({
                        'section_number': current_section['number'],
                        'section_path': current_section['path'],
                        'content': section_text,
                        'content_length': len(section_text),
                        'line_number': current_section['line']
                    })
            
            # Start new section
            level1 = match.group(1)
            level2 = match.group(2)
            level3 = match.group(3)
            content_after_number = match.group(4)
            
            # Skip if content after number looks like TOC (has dots and page numbers)
            if content_after_number and re.search(r'\.{3,}\s*\d+', content_after_number):
                continue
            
            # Build section path
            if level3:
                section_path = f"{level1}//{level2}//{level3}"
            elif level2:
                section_path = f"{level1}//{level2}"
            else:
                section_path = level1
            
            current_section = {
                'number': section_path,
                'path': section_path,
                'line': line_num
            }
            current_content = [content_after_number] if content_after_number else []
        else:
            # This line is continuation of current section content
            if current_section is not None:
                # Only add non-empty lines that aren't section headers or TOC entries
                if line_stripped and not re.match(r'^Shari[''\u2019]ah Standard No\\.', line_stripped, re.IGNORECASE):
                    # Skip if it's a TOC entry
                    if not is_toc_entry(line_stripped):
                        # Skip lines that are just "Appendices" or similar
                        if line_stripped not in ["Appendices", "Appendix", "(Revised Standard)"]:
                            # Skip numbered items with dots and page numbers (TOC format)
                            if not re.match(r'^\d+\.\s+.*\.{3,}\s*\d+$', line_stripped):
                                current_content.append(line_stripped)
    
    # Save last section
    if current_section is not None and current_content:
        section_text = '\n'.join(current_content).strip()
        if section_text:
            sections_data.append({
                'section_number': current_section['number'],
                'section_path': current_section['path'],
                'content': section_text,
                'content_length': len(section_text),
                'line_number': current_section['line']
            })
    
    # Filter out sections that are just TOC entries or have TOC content
    filtered_sections = []
    for section in sections_data:
        content = section['content']
        # Skip sections that are primarily TOC entries
        # Check if content contains multiple TOC patterns
        content_lines = content.split('\n')
        toc_lines = [line for line in content_lines if is_toc_entry(line.strip())]
        if len(content_lines) > 0 and len(toc_lines) > len(content_lines) * 0.5:  # More than 50% TOC lines
            continue
        # Skip sections where content is just a TOC entry
        if is_toc_entry(content.strip()):
            continue
        # Skip very short sections that look like TOC entries
        if len(content) < 100 and re.search(r'\.{3,}\s*\d+', content):
            continue
        filtered_sections.append(section)
    
    sections_df = pd.DataFrame(filtered_sections)
    print(f"\nExtracted {len(sections_df)} sections with content (filtered from {len(sections_data)} total)")
    
    return cleaned_text, sections_df

def main():
    txt_files = list(CLEAN_TEXT_DIR.glob("*.txt"))
    
    if not txt_files:
        print(f"No text files found in {CLEAN_TEXT_DIR}")
        return
    
    for txt_file in txt_files:
        if "cleaned" in txt_file.name:
            continue  # Skip already cleaned files
            
        print(f"\n{'='*60}")
        print(f"Processing: {txt_file.name}")
        print(f"{'='*60}")
        
        cleaned_text, sections_df = clean_text_and_extract_standards(txt_file)
        
        if cleaned_text is None:
            continue
        
        # Save cleaned text
        output_file = CLEAN_TEXT_DIR / f"{txt_file.stem}-cleaned.txt"
        output_file.write_text(cleaned_text, encoding="utf-8")
        print(f"\nSaved cleaned text to: {output_file}")
        
        # Save dataset
        dataset_path = DATASET_DIR / "standards_dataset.csv"
        sections_df.to_csv(dataset_path, index=False)
        print(f"Saved dataset to: {dataset_path}")
        print(f"Dataset shape: {sections_df.shape}")
        print(f"Total content length: {sections_df['content_length'].sum():,} characters")
        
        print(f"\nFirst few sections:")
        print(sections_df[['section_number', 'content_length']].head(10))
        print(f"\nSample content from first section:")
        if len(sections_df) > 0:
            print(sections_df.iloc[0]['content'][:200] + "...")

if __name__ == "__main__":
    main()

