"""
mongo_utils.py  – stub version

In the no-DB workflow we don’t want to install or connect to MongoDB.
`insert_doc` is kept so other modules can import it without changes.

If you later decide to use Mongo, replace this file with the original
implementation that creates a `MongoClient` and performs inserts.
"""


def insert_doc(*_, **__):
    """Dummy function that does nothing."""
    return
