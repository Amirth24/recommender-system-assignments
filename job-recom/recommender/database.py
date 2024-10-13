import os
import pymongoarrow.monkey
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1/")

__MONGO_CLIENT = None
__INITIALIZED = False


def init_db():
    global __MONGO_CLIENT, __INITIALIZED
    pymongoarrow.monkey.patch_all()
    __MONGO_CLIENT = MongoClient(MONGO_URI)

    print("Database Initialized")
    __INITIALIZED = True


def get_db():
    if not __INITIALIZED:
        raise Exception("Database Not Intialized")
    return __MONGO_CLIENT.sei
