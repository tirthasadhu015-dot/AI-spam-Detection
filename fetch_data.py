import pandas as pd
import io
import requests
import zipfile

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

try:
    print("Dataset download hocche, ektu opekkha koro...")
    
    # Zip file ta download kora hocche
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Specific-bhabe shudhu main data file-ta extract/read korchi
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='\t', names=['v1', 'v2'], encoding='utf-8')
    
    # Save to your project folder
    df.to_csv('spam.csv', index=False)
    print("\nSabaash! Dataset successfully downloaded as 'spam.csv'!")
    
    # Data Preview
    print("\nData Preview:")
    print(df.head())

except Exception as e:
    print(f"\nKono error hoyeche: {e}")