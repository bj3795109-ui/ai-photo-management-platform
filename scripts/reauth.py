import os
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/photospicker.mediaitems.readonly",
]

def main():
    if os.path.exists("token.json"):
        os.remove("token.json")
    
    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
    
    print("\nStarting local server on port 8085...", flush=True)
    creds = flow.run_local_server(
        host='localhost',
        port=8085,
        prompt='consent',
        access_type='offline',
        authorization_prompt_message='\n' + '='*80 + '\nOPEN THIS URL TO AUTHORIZE:\n{url}\n' + '='*80 + '\n'
    )
    
    with open("token.json", "w") as token_file:
        token_file.write(creds.to_json())
    print("\nSUCCESS: Token saved to token.json!", flush=True)

if __name__ == "__main__":
    main()
