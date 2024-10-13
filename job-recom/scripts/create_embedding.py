import os
import multiprocessing
import tqdm
import ollama
import pymongo


MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1")

client = pymongo.MongoClient(MONGO_URI)
db = client.sei
collection = db.jobs
requesting = []

print("Creating Embedding for job description")


def create_embed(job):
    desc = job["job_description"]
    job["desc_embedding"] = ollama.embeddings(
        model="nomic-embed-text", prompt=desc)
    return pymongo.UpdateOne({"_id": job["_id"]}, {
        "$set": {"desc_embedding": job["desc_embedding"]},
    })


fil_qu = {"desc_embedding": {"$exists": False}}
with multiprocessing.Pool() as pool:
    for i, result in enumerate(pool.imap_unordered(
            create_embed,
            tqdm.tqdm(
            collection.find(filter=fil_qu, projection={
                            "job_description": True}),
            total=collection.count_documents(fil_qu)))):
        requesting.append(result)
        if i % 100 == 0:
            result = collection.bulk_write(requesting)
            requesting = []


client.close()
