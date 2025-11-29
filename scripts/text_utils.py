"""
Text processing utilities for PDF extraction and cleaning.
"""
import re


def fix_double_characters(text, verbose=False):
    """
    Fix double character duplication issue from PDF extraction.
    Detects if text has pattern like "FFoorreewwoorrdd" and converts to "Foreword"
    
    Args:
        text: The text to process
        verbose: If True, print detection messages
        
    Returns:
        The fixed text with duplicate characters removed
    """
    if not text:
        return text
    
    # Check if text has significant double-character pattern
    # Sample first 3000 characters to detect the pattern
    sample = text[:3000] if len(text) > 3000 else text
    double_char_pattern = re.compile(r'([A-Za-z0-9])\1')
    matches = len(double_char_pattern.findall(sample))
    total_alnum = len(re.findall(r'[A-Za-z0-9]', sample))
    
    # If more than 20% of alphanumeric characters are part of double patterns, fix it
    if total_alnum > 30 and matches / total_alnum > 0.20:
        if verbose:
            print(f"  Detected double-character pattern ({matches}/{total_alnum} = {matches/total_alnum:.1%}), fixing...")
        
        # Remove duplicate characters: if we see "FF" or "oo", keep only one
        # Process character by character, removing consecutive duplicates of alphanumeric chars
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            
            # If this is alphanumeric and the next character is the same, skip the duplicate
            if char.isalnum() and i + 1 < len(text) and text[i + 1] == char:
                result.append(char)
                i += 2  # Skip both the current and duplicate
            else:
                result.append(char)
                i += 1
        
        return ''.join(result)
    return text


def fix_double_punctuation(text, verbose=False):
    """
    Fix double punctuation marks from PDF extraction.
    Fixes patterns like ",," -> ",", ".." -> ".", "::" -> ":", etc.
    
    Args:
        text: The text to process
        verbose: If True, print detection messages
        
    Returns:
        The fixed text with double punctuation removed
    """
    if not text:
        return text
    
    # Count double punctuation patterns in a sample
    sample = text[:3000] if len(text) > 3000 else text
    double_punct_patterns = [
        (r',,', ','),      # Double comma
        (r'\.\.', '.'),    # Double period (but preserve ellipsis ...)
        (r'::', ':'),      # Double colon
        (r';;', ';'),      # Double semicolon
        (r"''", "'"),      # Double apostrophe
        (r'``', '`'),      # Double backtick
        (r'\(\(', '('),    # Double opening parenthesis (escaped)
        (r'\)\)', ')'),
        (r"’’", "’"),
        # Note: // is NOT fixed - it's intentional section numbering (e.g., "3//11")
    ]
    
    total_fixes = 0
    fixed_text = text
    
    for pattern, replacement in double_punct_patterns:
        # Count occurrences
        matches = len(re.findall(pattern, sample))
        if matches > 0:
            # Fix all occurrences in the full text
            fixed_text = re.sub(pattern, replacement, fixed_text)
            total_fixes += matches
            if verbose:
                print(f"  Fixed {matches} instances of '{pattern}' -> '{replacement}'")
    
    # Special handling for double dashes: "--" might be intentional (em-dash)
    # But if it's clearly a mistake (like "well--being"), fix it
    # We'll be conservative and only fix if it's between words
    double_dash_matches = len(re.findall(r'\w--\w', sample))
    if double_dash_matches > 0:
        fixed_text = re.sub(r'(\w)--(\w)', r'\1-\2', fixed_text)
        total_fixes += double_dash_matches
        if verbose:
            print(f"  Fixed {double_dash_matches} instances of '--' between words -> '-'")
    
    if verbose and total_fixes > 0:
        print(f"  Total double punctuation fixes: {total_fixes}")
    
    return fixed_text


def clean_text(text, fix_doubles=True, fix_punctuation=True, verbose=False):
    """
    Clean extracted text from PDFs.
    
    Args:
        text: The raw text to clean
        fix_doubles: If True, fix double character duplication
        fix_punctuation: If True, fix double punctuation marks
        verbose: If True, print progress messages
        
    Returns:
        The cleaned text
    """
    # Fix double character duplication first
    if fix_doubles:
        text = fix_double_characters(text, verbose=verbose)
    
    # Fix double punctuation marks
    if fix_punctuation:
        text = fix_double_punctuation(text, verbose=verbose)
    
    # Basic cleaning: remove multiple line breaks, page numbers, etc.
    text = text.replace("\r", "\n")
    # Remove excessive line breaks
    text = re.sub(r'\n{2,}', '\n\n', text)
    # Remove page numbers like "Page X of Y"
    text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text, flags=re.IGNORECASE)
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()


def remove_metadata(text, verbose=False):
    """
    Remove metadata, table of contents, prefaces, and other non-standard content.
    Keeps only content starting from the first standard.
    
    Args:
        text: The text to clean
        verbose: If True, print what's being removed
        
    Returns:
        The cleaned text with only standard content
    """
    if not text:
        return text
    
    lines = text.splitlines()
    cleaned_lines = []
    found_first_standard = False
    in_toc = False
    in_preface = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        # Find first standard - keep everything from here
        if not found_first_standard:
            if re.search(r"Shari'ah Standard No\.\s*\((\d+)\)", line_stripped, re.IGNORECASE):
                found_first_standard = True
                if verbose:
                    print(f"  Found first standard at line {i+1}: {line_stripped[:80]}")
                cleaned_lines.append(line)
                i += 1
                continue
            else:
                i += 1
                continue
        
        # Now we're past the first standard - clean up metadata
        
        # Skip table of contents sections
        if line_stripped == "Table of Contents" or line_stripped == "Contents":
            in_toc = True
            if verbose:
                print(f"  Removing TOC at line {i+1}")
            i += 1
            continue
        
        if in_toc:
            # Skip TOC entries until we find a standard or section
            if re.search(r"Shari'ah Standard No\.|^\d+//", line_stripped, re.IGNORECASE):
                in_toc = False
                cleaned_lines.append(line)
            elif line_stripped == "Subject Page" or re.match(r'^[A-Z][^:]*\.{3,}\s*\d+$', line_stripped):
                # Skip TOC entry lines
                pass
            elif not line_stripped:
                # Empty line in TOC
                pass
            else:
                # End of TOC
                in_toc = False
                cleaned_lines.append(line)
            i += 1
            continue
        
        # Skip "Subject Page" lines
        if line_stripped == "Subject Page":
            i += 1
            continue
        
        # Skip Preface sections (with page numbers like "Preface ......................... 51")
        if re.match(r'^Preface\s*\.{3,}\s*\d+$', line_stripped):
            in_preface = True
            if verbose:
                print(f"  Removing Preface at line {i+1}")
            i += 1
            # Skip until Statement of the Standard or section number
            while i < len(lines):
                next_line = lines[i].strip()
                if re.search(r"Statement of the Standard|^\d+//", next_line, re.IGNORECASE):
                    in_preface = False
                    break
                i += 1
            continue
        
        # Skip standalone "Preface" headers (not part of standard content)
        if line_stripped == "Preface" and i + 1 < len(lines):
            next_line = lines[i+1].strip()
            # If next line is empty or another header, skip this preface
            if not next_line or next_line in ["Contents", "Subject Page"]:
                in_preface = True
                if verbose:
                    print(f"  Removing Preface header at line {i+1}")
                i += 1
                while i < len(lines) and not re.search(r"Statement of the Standard|^\d+//|Shari'ah Standard No\.", lines[i].strip(), re.IGNORECASE):
                    i += 1
                continue
        
        # Skip page number references (lines ending with dots and numbers)
        if re.match(r'^[A-Z][^:]*\.{3,}\s*\d+$', line_stripped):
            i += 1
            continue
        
        # Keep all other lines (including standard content, sections, etc.)
        cleaned_lines.append(line)
        i += 1
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Final cleanup: remove excessive blank lines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    if verbose:
        original_len = len(text)
        cleaned_len = len(cleaned_text)
        removed = original_len - cleaned_len
        pct = (removed / original_len * 100) if original_len > 0 else 0
        print(f"  Removed {removed:,} characters ({pct:.1f}%)")
        print(f"  Final size: {cleaned_len:,} characters, {len(cleaned_lines):,} lines")
    
    return cleaned_text.strip()

