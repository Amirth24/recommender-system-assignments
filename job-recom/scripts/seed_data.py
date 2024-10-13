import re
import os
import pymongo
import json
import tqdm
from pymongo import InsertOne

import spacy
from spacy.matcher import Matcher


MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Define patterns to match experience-related phrases
experience_patterns = [
    # Match a number followed by "year(s)" or "yr(s)" followed by "of experience"
    [{"LIKE_NUM": True}, {"LOWER": {"in": ["years", "year", "yrs", "yr"]}},
        {"LOWER": "of"}, {"LOWER": "experience"}],
    # Match a range (e.g., "4-5") followed by "years of experience"
    [{"LIKE_NUM": True}, {"TEXT": "-"}, {"LIKE_NUM": True},
        {"LOWER": {"in": ["years", "yrs"]}}, {"LOWER": "of"}, {"LOWER": "experience"}],
    # Match a number followed by "year(s)" or "yr(s)" without "of experience"
    [{"LIKE_NUM": True}, {"LOWER": {"in": ["years", "year", "yrs", "yr"]}}],
]

# Add patterns to the matcher
for pattern in experience_patterns:
    matcher.add("EXPERIENCE", [pattern])


def extract_years(experience_text):
    pattern = r'(\d+)[-\s]*(\d+)?\s*(years?|yrs?)'
    match = re.search(pattern, experience_text, re.IGNORECASE)

    if match:
        # If it's a range (X-Y), capture both X and Y
        if match.group(2):
            # Return both numbers in the range
            return int(match.group(1)), int(match.group(2))
        else:
            # Return the single number twice (for consistency)
            return int(match.group(1)), int(match.group(1))
    return 0, 0


# Function to extract experience using spaCy matcher
def extract_experience(job_desc):
    doc = nlp(job_desc)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        min_yr, max_yr = extract_years(span.text)
        if min_yr > max_yr or min_yr >= 15 or max_yr >= 15:
            continue
        return min_yr, max_yr
    return 0, 0


client = pymongo.MongoClient(MONGO_URI)
db = client.sei
collection = db.jobs
requesting = []

with open(r"dataset/job_dataset.json") as f:
    for jsonObj in tqdm.tqdm(f, total=29998):
        try:
            myDict = json.loads(jsonObj)
            myDict["_id"] = myDict["uniq_id"]
            del myDict["uniq_id"]
            myDict['min_exp_yrs'], myDict['min_exp_yrs'] = extract_experience(
                myDict['job_description'])
            requesting.append(InsertOne(myDict))
        except json.JSONDecodeError:
            pass

result = collection.bulk_write(requesting)
client.close()
