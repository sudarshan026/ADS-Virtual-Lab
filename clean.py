import os
import re

print("Starting CSS cleanup...")
base_dir = r"D:\adsvl"

for root, dirs, files in os.walk(base_dir):
    if 'venv' in root or '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py') or file.endswith('.html'):
            fpath = os.path.join(root, file)
            # skip the global theme itself!
            if file == 'theme.py' or file == 'clean.py':
                continue
            
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "<style>" in content or "<style " in content:
                    new_content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
                    if new_content != content:
                        with open(fpath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Removed custom CSS from {os.path.relpath(fpath, base_dir)}")
            except Exception as e:
                pass
print("Done.")