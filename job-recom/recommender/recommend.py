import numpy as np
import ollama
import pandas as pd
from pymongoarrow.api import Schema
import pyarrow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .database import get_db

__DESCRIPTION = None
__DESCRIPTION_VEC = None
__ROLES_VEC = None

__DESCRIPTION_SIM = None

__ROLES_MAT = None
__DESCRIPTION_MAT = None

__JOB_SERIES = None
INITIALIZED = False


def init_recommendation():
    global __DESCRIPTION, INITIALIZED, __DESCRIPTION_SIM, __ROLES_SIM, __ROLES_MAT, __DESCRIPTION_MAT, __JOB_SERIES

    print("Loading Embeddings")
    __DESCRIPTION = pd.read_parquet("embedding.parquet", engine="pyarrow")

    if __DESCRIPTION_MAT is None:
        __DESCRIPTION_MAT = np.array(
            [arr[1]['desc_embedding'] for arr in __DESCRIPTION.iterrows()])
        __JOB_SERIES = __DESCRIPTION.index
    INITIALIZED = True


def init_recommendation_old():
    global __DESCRIPTION_VEC, __ROLES_VEC, __DESCRIPTION_SIM, __ROLES_SIM, __ROLES_MAT, __DESCRIPTION_MAT, __JOB_SERIES

    jobs = get_db().jobs

    job_df = jobs.find_pandas_all(
        {},
        schema=Schema({
            '_id': pyarrow.string(),
            'job_title': pyarrow.string(),
            'company_name': pyarrow.string(),
            'company_description': pyarrow.string(),
            'job_description': pyarrow.string(),
            'job_type': pyarrow.string(),
            'category': pyarrow.string()
        }),
        limit=1000,
    )
    job_df.set_index('_id', inplace=True)

    # Feature Extraction

    description = job_df["job_description"].tolist()

    __DESCRIPTION_VEC = TfidfVectorizer(stop_words="english")
    __DESCRIPTION_MAT = __DESCRIPTION_VEC.fit_transform(description)
    __DESCRIPTION_SIM = cosine_similarity(__DESCRIPTION_MAT)

    __ROLES_VEC = TfidfVectorizer(stop_words="english")
    __ROLES_MAT = __ROLES_VEC.fit_transform(
        job_df['job_title'].apply(lambda x: x.lower()).tolist())

    __JOB_SERIES = job_df.index


def recommend_jobs_for_roles(description, k=5, j=10, m=200):

    desc_emb = np.array(ollama.embeddings(
        model="nomic-embed-text", prompt=description)['embedding'])

    jd_sim = cosine_similarity(desc_emb.reshape(1, -1), __DESCRIPTION_MAT)

    return [
        __JOB_SERIES[x]
        for x in jd_sim.squeeze().argsort()[-1:-m:-1]
    ]


def get_recommendation(min_experience, about_user):

    ids = recommend_jobs_for_roles(about_user)

    jobs = get_db().jobs.find_pandas_all({
        "_id": {"$in": ids},
        "min_exp_yrs": {"$lte": min_experience},
    }).sort_values("min_exp_yrs", ascending=False)

    return jobs
