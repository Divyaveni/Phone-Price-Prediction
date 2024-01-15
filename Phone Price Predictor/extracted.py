import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('phone_data_processed.csv')

# Extract RAM, ROM, Display, Camera, and Battery from 'Features' column
df['RAM(GB)'] = df['Features'].apply(lambda x: re.search(r"\'(\d+)\'\s*:\s*\'GB\s*RAM", str(x)).group(1) if re.search(r"\'(\d+)\'\s*:\s*\'GB\s*RAM", str(x)) else None)
df['ROM(GB)'] = df['Features'].apply(lambda x: re.search(r"\'(\d+)\'\s*:\s*\'GB\s*ROM", str(x)).group(1) if re.search(r"\'(\d+)\'\s*:\s*\'GB\s*ROM", str(x)) else (re.search(r"(\d+)\s*GB", str(x)).group(1) if re.search(r"(\d+)\s*GB", str(x)) else None))
df['Display(inch)'] = df['Features'].apply(lambda x: re.search(r"(\d+\.\d+)\s*inch", str(x)).group(1) if (re.search(r"(\d+\.\d+)\s*inch", str(x)) and pd.notnull(x)) else None)
df['Camera(MP)'] = df['Features'].apply(lambda x: re.search(r"(\d+)\s*MP", str(x)).group(1) if (re.search(r"(\d+)\s*MP", str(x)) and pd.notnull(x)) else None)
df['Battery(mAh)'] = df['Features'].apply(lambda x: re.search(r"(\d+)\s*mAh", str(x)).group(1) if (re.search(r"(\d+)\s*mAh", str(x)) and pd.notnull(x)) else None)

# Remove 'Features' column
df = df.drop('Features', axis=1)

# Remove rupees symbol from 'Price' column
df['Price(rupees)'] = df['Price'].replace('[\₹,]', '', regex=True).astype(float)
df = df.drop('Price', axis=1)

# Remove %off from 'Offer' column and rename it to 'Discount(%off)'
df['Offer'] = df['Offer'].replace('% off', '', regex=True).astype(float)
df = df.rename(columns={'Offer': 'Discount(%off)'})

# Extract features from Brand_Name
def extract_model(x):
    tokens = x.split(' ')
    if 'G' in x:
        return ' '.join(tokens[0:2]).strip() if len(tokens) >= 2 else ''
    else:
        return ' '.join(tokens[0:2]).strip() if len(tokens) >= 2 else ''

def extract_generation(x):
    match = re.search(r'(?:^|\s)(\d+)G(?=\s|$)', x)
    return match.group(1) if match else ''

def extract_color(x):
    match = re.search(r'\(([^)]+)', x)
    return match.group(1).split(',')[0].strip() if match else ''

df['Brand_Model'] = df['Brand_Name'].apply(lambda x: extract_model(x.split('(')[0]).strip())
df['Generation(G)'] = df['Brand_Name'].apply(extract_generation)
df['Color'] = df['Brand_Name'].apply(extract_color)

# Drop the original Brand_Name column
df = df.drop(['Brand_Name'], axis=1)

# Remove the generation from the 'Brand_Model' column
df['Brand_Model'] = df.apply(lambda row: row['Brand_Model'].replace(row['Generation(G)'], '').strip(), axis=1)

# Reorder columns as per your request
df = df[['Brand_Model', 'Generation(G)', 'Color', 'Ratings', 'RAM(GB)', 'ROM(GB)', 'Display(inch)', 'Camera(MP)', 'Battery(mAh)', 'Reviews', 'Discount(%off)', 'Price(rupees)']]

# Save the updated DataFrame to a new CSV file
df.to_csv('data_updated.csv', index=False)

# Display the updated DataFrame
print(df.head())


# Split 'Brand_Model' into 'Brand' and 'Model'
df[['Brand', 'Model']] = df['Brand_Model'].str.split(' ', 1, expand=True)

# Drop the original 'Brand_Model' column
df = df.drop(['Brand_Model'], axis=1)
df = df.drop(['Model'],axis=1)

# Reorder columns as needed
df = df[['Brand', 'Generation(G)', 'Color', 'Ratings', 'RAM(GB)', 'ROM(GB)', 'Display(inch)', 'Camera(MP)', 'Battery(mAh)', 'Reviews', 'Discount(%off)', 'Price(rupees)']]

# Save the updated DataFrame to a new CSV file
df.to_csv('data_updated_split.csv', index=False)

# Display the updated DataFrame
print(df.head())
