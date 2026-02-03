import os
from dotenv import load_dotenv
from huggingface_hub import login

# Absolute import from your sister folder 'utils'
# This works because of your __init__.py files and running from root
from my_app.utils.logger import log 

# Locate the .env file inside this same 'config' folder
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

def authenticate_hf():
    """
    Authenticates to Hugging Face using the token stored in .env
    """
    # Look for HF_TOKEN in your .env file
    token = os.getenv("HF_TOKEN")
    
    if token:
        try:
            # add_to_git_credential=False keeps your local git clean
            login(token=token, add_to_git_credential=False)
            
            # Use 'log' (which matches your logger.py variable name)
            log.info("Hugging Face Authentication Successful.")
            
        except Exception as e:
            log.error(f"Failed to authenticate with Hugging Face: {e}")
    else:
        log.warning("HF_TOKEN not found in .env. Gated models like Llama may not load.")

# This ensures authentication runs as soon as this file is imported
if __name__ == "__main__":
    authenticate_hf()