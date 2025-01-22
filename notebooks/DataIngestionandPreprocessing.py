pip install telethon
from telethon import TelegramClient
import pandas as pd
import json
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Define your API credentials (you need to get these from Telegram API)
api_id = 'your_api_id'
api_hash = 'your_api_hash'
phone_number = 'your_phone_number'

# Create a client
client = TelegramClient('session_name', api_id, api_hash)

# List of channels to scrape
channels = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5']

# Function to get messages from a channel
async def fetch_messages(channel):
    messages = []
    async for message in client.iter_messages(channel):
        messages.append({
            'message_id': message.id,
            'sender': message.sender_id,
            'timestamp': message.date,
            'message': message.text
        })
    return messages

async def main():
    await client.start(phone_number)
    all_messages = []
    for channel in channels:
        messages = await fetch_messages(channel)
        all_messages.extend(messages)

    # Convert to DataFrame for better processing
    df = pd.DataFrame(all_messages)
    df.to_csv('telegram_messages.csv', index=False)
    print("Messages fetched and saved.")

# Run the async client
client.loop.run_until_complete(main())



# Preprocessing function
def preprocess_text(text):
    # Remove any unwanted characters (e.g., links, special characters)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^አ-፯a-zA-Z0-9፡።]", " ", text)  # Remove non-Amharic characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df['processed_message'] = df['message'].apply(preprocess_text)

# Tokenization example (for Amharic)
def tokenize_amharic(text):
    return word_tokenize(text)

df['tokenized_message'] = df['processed_message'].apply(tokenize_amharic)

df.to_csv('processed_telegram_messages.csv', index=False)
print("Data Preprocessing Complete")
