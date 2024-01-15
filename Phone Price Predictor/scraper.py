import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import re

# Function to clean up the price and use the Indian numbering system
def clean_price(price):
    # Check if '₹' is present in the price string
    if '₹' in price:
        # Remove non-numeric characters and convert to Indian numbering system
        cleaned_price = re.sub(r'[^\d.]', '', price.split('₹')[1])
        return f"₹{cleaned_price}"
    else:
        return None  # Return None if '₹' is not present

# Function to extract numeric values from reviews
def extract_numeric_reviews(review):
    match = re.match(r'([\d,]+)\sRatings', review)
    if match:
        return int(match.group(1).replace(',', ''))
    else:
        return None

# Function to extract features
def extract_features(features):
    feature_dict = {}
    feature_list = features.split('|')
    for feature in feature_list:
        key_value = feature.strip().split(maxsplit=1)
        if len(key_value) == 2:
            key = key_value[0].lower()
            value = key_value[1].strip()
            feature_dict[key] = value
            # Extracting RAM, ROM, Display, Camera, Processor, and Warranty separately
            if key in ['ram', 'rom', 'display', 'camera', 'processor', 'warranty']:
                feature_dict[key] = value
    return feature_dict

# Function to scrape data from a single page
def scrape_page(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")

    # Extracting relevant information
    Brand_Name = soup.find_all('div', class_='_4rR01T')
    Ratings = soup.find_all('div', class_='_3LWZlK')
    Reviews = soup.find_all('span', class_='_2_R_DZ')
    Price = soup.find_all('div', class_='_25b18c')
    Storage_Disp_Cam_Bat_Proc_Warranty = soup.find_all('ul', class_='_1xgFaf')
    Offer = soup.find_all('div', class_='_3Ay6Sb')

    # Create a list to store data
    data = []

    # Iterate through the extracted elements and store in the data list
    for i in range(len(Brand_Name)):
        phone_data = {
            'Brand_Name': Brand_Name[i].text,
            'Ratings': Ratings[i].text if i < len(Ratings) else None,
            'Reviews': extract_numeric_reviews(Reviews[i].text) if i < len(Reviews) else None,
            'Price': clean_price(Price[i].text) if i < len(Price) else None,
            'Features': extract_features(Storage_Disp_Cam_Bat_Proc_Warranty[i].text) if i < len(Storage_Disp_Cam_Bat_Proc_Warranty) else None,
            'Offer': Offer[i].text if i < len(Offer) else None
        }
        data.append(phone_data)

    return data

# Function to scrape data from multiple pages
def scrape_multiple_pages(base_url, num_pages):
    all_data = []

    for page in range(1, num_pages + 1):
        url = f"{base_url}&page={page}"
        data = scrape_page(url)
        all_data.extend(data)

    return all_data

# Define the base URL
base_url = "https://www.flipkart.com/search?q=phones"

# Specify the number of pages to scrape (in this case, 24 pages)
num_pages = 24

# Scrape data from multiple pages
all_phone_data = scrape_multiple_pages(base_url, num_pages)

# Convert the data to a DataFrame for further analysis
df = pd.DataFrame(all_phone_data)

# Save the DataFrame to a CSV file
df.to_csv('phone_data_processed.csv', index=False)
