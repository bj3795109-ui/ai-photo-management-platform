import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Integer, Boolean, Index, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from db.database import Base

CLIP_DIM = 512


def gen_uuid():
    return str(uuid.uuid4())


class Photo(Base):
    __tablename__ = "photos"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    source_path = Column(String, nullable=False)
    storage_backend = Column(String, nullable=False, default="local")
    storage_ref = Column(String, nullable=False)
    hash = Column(String, index=True)
    timestamp = Column(DateTime, nullable=True)
    gps_lat = Column(Float, nullable=True)
    gps_lng = Column(Float, nullable=True)
    embedding = Column(Vector(CLIP_DIM), nullable=True)
    faces_processed = Column(Boolean, default=False)
    dedup_processed = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class Duplicate(Base):
    __tablename__ = "duplicates"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    photo_id = Column(UUID(as_uuid=False), ForeignKey("photos.id"), nullable=False)
    duplicate_of_id = Column(UUID(as_uuid=False), ForeignKey("photos.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)


class FaceCluster(Base):
    __tablename__ = "face_clusters"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    label = Column(String, nullable=True)


class Face(Base):
    __tablename__ = "faces"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    photo_id = Column(UUID(as_uuid=False), ForeignKey("photos.id"), nullable=False)
    bbox = Column(JSONB, nullable=False)
    embedding = Column(Vector(128), nullable=True)
    cluster_id = Column(UUID(as_uuid=False), ForeignKey("face_clusters.id"), nullable=True)


class Category(Base):
    __tablename__ = "categories"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    photo_id = Column(UUID(as_uuid=False), ForeignKey("photos.id"), nullable=False)
    category = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)


class JobStatus(Base):
    __tablename__ = "job_status"

    job_name = Column(String, primary_key=True)  # e.g. "dedup", "faces_cluster"
    processed = Column(Integer, default=0)
    total = Column(Integer, default=0)
    status = Column(String, default="idle")  # idle | running | done
    extra_info = Column(String, nullable=True)  # e.g. "1234 duplicates found so far"
    updated_at = Column(DateTime, nullable=True)


Index(
    "ix_photos_embedding_cosine",
    Photo.embedding,
    postgresql_using="ivfflat",
    postgresql_with={"lists": 100},
    postgresql_ops={"embedding": "vector_cosine_ops"}
)