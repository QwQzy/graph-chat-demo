from motor.motor_asyncio import AsyncIOMotorClient


def get_mongo_client():
    return AsyncIOMotorClient(host="localhost", port=27017)