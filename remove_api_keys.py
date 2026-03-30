import os
import shutil
import re
import math
import subprocess
from pathlib import Path

# --- Configuration ---
# You can tweak these values. A higher entropy threshold is stricter.
ENTROPY_THRESHOLD = 3.5 
MIN_KEY_LENGTH = 20
# Common keywords that often precede secrets
SECRET_KEYWORDS = [
    'key', 'api', 'token', 'secret', 'password', 'auth', 'client_id', 'client_secret'
]

def calculate_entropy(s):
    """[REDACTED_BY_SCRIPT]"""
    if not s:
        return 0
    # Get the frequency of each character in the string
    char_counts = {c: s.count(c) for c in set(s)}
    # Calculate the entropy
    entropy = -sum((count / len(s)) * math.log2(count / len(s)) for count in char_counts.values())
    return entropy

def redact_secrets_in_file(file_path):
    """[REDACTED_BY_SCRIPT]"""
    redactions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for i, line in enumerate(lines):
            original_line = line
            
            # 1. Find strings in quotes and check their entropy
            # This regex finds content within single or double quotes
            string_literals = re.findall(r'["\'](.*?)["\']', line)
            for s in string_literals:
                if len(s) >= MIN_KEY_LENGTH and calculate_entropy(s) > ENTROPY_THRESHOLD:
                    line = line.replace(s, "[REDACTED_BY_SCRIPT]")
                    redactions.append(f"[REDACTED_BY_SCRIPT]")

            # 2. Find variables assigned using common keywords
            # This regex looks for patterns like `api_key = "..."`
            for keyword in SECRET_KEYWORDS:
                match = re.search(fr'[REDACTED_BY_SCRIPT]"\'](.*?)["\']', line, re.IGNORECASE)
                if match:
                    secret_value = match.group(1)
                    if secret_value != "[REDACTED_BY_SCRIPT]": # Avoid double-redacting
                        line = line.replace(secret_value, "[REDACTED_BY_SCRIPT]")
                        redactions.append(f"[REDACTED_BY_SCRIPT]'{keyword}'[REDACTED_BY_SCRIPT]")
            
            new_lines.append(line)

        # If any changes were made, overwrite the file with the redacted content
        if redactions:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

    return redactions


def main():
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    source_dir = r"E:\Wisteria"
    dest_dir = r"E:\Wisteria_Clean"

    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # --- Safety Checks ---
    if not source_path.is_dir():
        print(f"Error: Source directory '{source_dir}' not found.")
        return
    if dest_path.exists():
        print(f"[REDACTED_BY_SCRIPT]'{dest_dir}'[REDACTED_BY_SCRIPT]")
        return

    def git_aware_ignore(src, names):
        # By default ignore everything except explicitly allowed extensions
        ignored = set()
        allowed_extensions = {'.py', '.sh', '.txt', '.md', '.json', '.yaml', '.yml'}
        
        # Don't ignore directories, we need to traverse them
        for name in names:
            full_path = os.path.join(src, name)
            if not os.path.isdir(full_path):
                # If it's a file, ignore it if its extension is not in the allowed list
                ext = Path(name).suffix.lower()
                if ext not in allowed_extensions:
                    if ext != '': # allow extension-less files like dockerfile etc but be careful
                         ignored.add(name)

        try:
            paths_to_check = [os.path.join(src, name) for name in names]
            result = subprocess.run(
                ['git', 'check-ignore', '--stdin'],
                input='\n'.join(paths_to_check),
                text=True,
                capture_output=True,
                cwd=source_dir
            )
            if result.stdout:
                ignored_paths = result.stdout.splitlines()
                ignored.update([os.path.basename(p) for p in ignored_paths])
        except Exception:
            pass
        
        # Standard ignores and .git directory
        default_ignores = shutil.ignore_patterns('*.pyc', '__pycache__', 'venv', '.venv', '.git', '*.csv', '*.parquet', '*.geoparquet', '*.xlsx', '*.png', '*.jpg', '*.h5', '*.pkl', '*.joblib', 'chromedriver-win64')(src, names)
        ignored.update(default_ignores)
        return list(ignored)

    print(f"[REDACTED_BY_SCRIPT]'{source_path}' to '{dest_path}'...")
    try:
        # Copy the directory tree using the git-aware ignore function
        shutil.copytree(source_path, dest_path, ignore=git_aware_ignore)
        print("    Copy complete.")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return
        
    print("[REDACTED_BY_SCRIPT]")
    all_redactions = {}
    # Walk through the new directory
    for root, _, files in os.walk(dest_path):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                redactions_found = redact_secrets_in_file(file_path)
                if redactions_found:
                    # Store results for the final report
                    relative_path = file_path.relative_to(dest_path)
                    all_redactions[str(relative_path)] = redactions_found

    print("[3] Scan complete.")
    print("\n--- FINAL REPORT ---")
    if not all_redactions:
        print("[REDACTED_BY_SCRIPT]")
    else:
        print(f"[REDACTED_BY_SCRIPT]")
        for file, details in all_redactions.items():
            print(f"\nFile: {file}")
            for detail in details:
                print(detail)

    print("\n--- !!! IMPORTANT !!! ---")
    print("[REDACTED_BY_SCRIPT]")
    print(dest_dir)
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")

main()