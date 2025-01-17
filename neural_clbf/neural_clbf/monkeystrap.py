import distutils
import types
import os
import re
import shutil

def monkey_patch_tqdm():
    """
    Monkey patch tqdm to fix the issue of tqdm not being imported correctly in some environments.
    This function will only patch the file if it hasn't been patched already.
    """
    def find_venv_dir():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != os.path.dirname(current_dir):  # Stop at root directory
            for item in os.listdir(current_dir):
                if item in ['venv', '.venv', 'myenv'] and os.path.isdir(os.path.join(current_dir, item)):
                    return os.path.join(current_dir, item)
            current_dir = os.path.dirname(current_dir)
        return None

    venv_dir = find_venv_dir()
    if venv_dir is None:
        print("[TQDM PATCH] [WARNING] Virtual environment directory not found.")
        return

    torchmetrics_dir = os.path.join(venv_dir, 'Lib', 'site-packages', 'torchmetrics')
    bert_file_path = os.path.join(torchmetrics_dir, 'functional', 'text', 'bert.py')

    if not os.path.exists(bert_file_path):
        print(f"[TQDM PATCH] [WARNING] bert.py file not found at {bert_file_path}")
        return

    with open(bert_file_path, 'r') as file:
        content = file.read()

    new_function = """
def _get_progress_bar(dataloader: DataLoader, verbose: bool = False) -> Union[DataLoader, tqdm.tqdm]:
    \"\"\"Helper function returning either the dataloader itself when `verbose = False`, or it wraps the dataloader with
    `tqdm.tqdm`, when `verbose = True` to display a progress bar during the embeddings calculation.\"\"\"
    return tqdm.tqdm(dataloader) if verbose else dataloader
"""

    pattern = re.compile(r'def _get_progress_bar.*?(?=\n\n)', re.DOTALL)
    if pattern.search(content) is None:
        print("[TQDM PATCH] [INFO] The file has already been patched or the function to be patched is not found.")
        return

    new_content = pattern.sub(new_function.strip(), content)

    if new_content != content:
        # Create a backup of the original file
        backup_file_path = bert_file_path + '.bak'
        try:
            shutil.copy2(bert_file_path, backup_file_path)
            
            with open(bert_file_path, 'w') as file:
                file.write(new_content)
            print(f"[TQDM PATCH] [INFO] Torchmetrics tqdm patch applied successfully to {bert_file_path}")
        except Exception as e:
            print(f"[TQDM PATCH] [ERROR] An error occurred while patching: {str(e)}")
            if os.path.exists(backup_file_path):
                print("[TQDM PATCH] [INFO] Restoring the original file from backup...")
                shutil.copy2(backup_file_path, bert_file_path)
                print("[TQDM PATCH] [INFO] Original file restored.")
        finally:
            if os.path.exists(backup_file_path):
                os.remove(backup_file_path)
    else:
        print("[TQDM PATCH] [INFO] No changes were necessary. The file is already patched.")

def monkey_patch_distutils():
    """
    Attaches a 'version' attribute to the distutils module if it doesn't exist.
    This is a workaround for the AttributeError in PyTorch's tensorboard import.
    """
    if (not hasattr(distutils, 'version')):
        version_module = types.ModuleType('version')
        
        class LooseVersion:
            def __init__(self, vstring):
                self.vstring = vstring
                self.version = tuple(map(int, self.vstring.split('.')))
            
            def __str__(self):
                return self.vstring
            
            def __repr__(self):
                return f"LooseVersion ('{self.vstring}')"
            
            def __lt__(self, other):
                return self.version < other.version
            
            def __le__(self, other):
                return self.version <= other.version
            
            def __eq__(self, other):
                return self.version == other.version
            
            def __ge__(self, other):
                return self.version >= other.version
            
            def __gt__(self, other):
                return self.version > other.version
            
            def __ne__(self, other):
                return self.version != other.version

        version_module.LooseVersion = LooseVersion
        distutils.version = version_module
        print("[DISTUTILS PATCH] [INFO] Distutils version patch applied successfully.")
    else:
        print("[DISTUTILS PATCH] [INFO] Distutils version patch already applied.")
