from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["stylesync"]

def get_collection(company_id):
    """
    Returns a MongoDB collection object based on company_id.
    MongoDB will create the collection if it does not exist.
    """
    return db[f"company{company_id}"]
