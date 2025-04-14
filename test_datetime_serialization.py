import unittest
import json
from datetime import datetime

from src.tools.memory import Entity, Relation, JournalRecord, RecordType


class TestDatetimeSerialization(unittest.TestCase):
    def test_entity_serialization(self):
        """Test that Entity objects can be serialized to JSON."""
        # Create an entity with datetime fields
        entity = Entity(
            name="test-entity",
            entityType="test",
            observations=["Test observation"],
            created_at=datetime(2023, 1, 1, 12, 0, 0),
            updated_at=datetime(2023, 1, 2, 12, 0, 0)
        )
        
        # Serialize to JSON
        json_str = entity.model_dump_json()
        
        # Should not raise an exception
        data = json.loads(json_str)
        
        # Verify datetime fields are serialized as ISO format strings
        self.assertEqual(data["created_at"], "2023-01-01T12:00:00")
        self.assertEqual(data["updated_at"], "2023-01-02T12:00:00")
    
    def test_relation_serialization(self):
        """Test that Relation objects can be serialized to JSON."""
        # Create a relation with datetime fields
        relation = Relation(
            from_entity="entity1",
            to_entity="entity2",
            relation_type="connects-to",
            created_at=datetime(2023, 1, 1, 12, 0, 0)
        )
        
        # Serialize to JSON
        json_str = relation.model_dump_json()
        
        # Should not raise an exception
        data = json.loads(json_str)
        
        # Verify datetime fields are serialized as ISO format strings
        self.assertEqual(data["created_at"], "2023-01-01T12:00:00")
    
    def test_journal_record_serialization(self):
        """Test that JournalRecord objects can be serialized to JSON."""
        # Create a journal record with datetime fields
        record = JournalRecord(
            record_type=RecordType.ENTITY_CREATE,
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            data={"name": "test-entity", "entityType": "test"}
        )
        
        # Serialize to JSON
        json_str = record.model_dump_json()
        
        # Should not raise an exception
        data = json.loads(json_str)
        
        # Verify datetime fields are serialized as ISO format strings
        self.assertEqual(data["timestamp"], "2023-01-01T12:00:00")
        self.assertEqual(data["record_type"], "entity_create")


if __name__ == "__main__":
    unittest.main() 