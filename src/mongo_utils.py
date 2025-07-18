from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME, COLL_NAME

client = None
collection = None

def get_collection():
    global client, collection
    if MONGO_URI and collection is None:
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLL_NAME]
    return collection


def insert_doc(doc: dict):
    coll = get_collection()
    if coll:
        coll.insert_one(doc)
