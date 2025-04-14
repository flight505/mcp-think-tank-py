#!/usr/bin/env python3
"""
Simple direct test of JSON serialization for various classes.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import the classes we want to test
from src.tools.memory import Entity, Relation, JournalRecord, RecordType
from src.tools.thinkentity import ThinkEntity
from src.tools.think import Thought

def test_entity_serialization():
    """Test Entity serialization"""
    print("Testing Entity serialization...")
    entity = Entity(
        name="test-entity",
        entityType="test",
        observations=["Test observation"],
        created_at=datetime(2023, 1, 1, 12, 0, 0),
        updated_at=datetime(2023, 1, 2, 12, 0, 0)
    )
    
    # Test model_dump_json
    try:
        json_str = entity.model_dump_json()
        data = json.loads(json_str)
        print(f"  model_dump_json(): SUCCESS")
        print(f"  created_at: {data['created_at']}")
        print(f"  updated_at: {data['updated_at']}")
    except Exception as e:
        print(f"  model_dump_json(): FAILED - {e}")
    
    # Test model_dump + json.dumps
    try:
        data = entity.model_dump()
        json_str = json.dumps(data)
        print(f"  model_dump() + json.dumps(): SUCCESS")
    except Exception as e:
        print(f"  model_dump() + json.dumps(): FAILED - {e}")

def test_relation_serialization():
    """Test Relation serialization"""
    print("\nTesting Relation serialization...")
    relation = Relation(
        from_entity="entity1",
        to_entity="entity2",
        relation_type="connects-to",
        created_at=datetime(2023, 1, 1, 12, 0, 0)
    )
    
    # Test model_dump_json
    try:
        json_str = relation.model_dump_json()
        data = json.loads(json_str)
        print(f"  model_dump_json(): SUCCESS")
        print(f"  created_at: {data['created_at']}")
    except Exception as e:
        print(f"  model_dump_json(): FAILED - {e}")
    
    # Test model_dump + json.dumps
    try:
        data = relation.model_dump()
        json_str = json.dumps(data)
        print(f"  model_dump() + json.dumps(): SUCCESS")
    except Exception as e:
        print(f"  model_dump() + json.dumps(): FAILED - {e}")

def test_journal_record_serialization():
    """Test JournalRecord serialization"""
    print("\nTesting JournalRecord serialization...")
    record = JournalRecord(
        record_type=RecordType.ENTITY_CREATE,
        timestamp=datetime(2023, 1, 1, 12, 0, 0),
        data={"name": "test-entity", "entityType": "test"}
    )
    
    # Test model_dump_json
    try:
        json_str = record.model_dump_json()
        data = json.loads(json_str)
        print(f"  model_dump_json(): SUCCESS")
        print(f"  timestamp: {data['timestamp']}")
        print(f"  record_type: {data['record_type']}")
    except Exception as e:
        print(f"  model_dump_json(): FAILED - {e}")
    
    # Test model_dump + json.dumps
    try:
        data = record.model_dump()
        json_str = json.dumps(data)
        print(f"  model_dump() + json.dumps(): SUCCESS")
    except Exception as e:
        print(f"  model_dump() + json.dumps(): FAILED - {e}")

def test_think_entity_serialization():
    """Test ThinkEntity serialization"""
    print("\nTesting ThinkEntity serialization...")
    thought = Thought(
        structured_reasoning="Test reasoning",
        category="test",
        tags=["test1", "test2"]
    )
    think_entity = ThinkEntity(
        thought=thought,
        created_at=datetime(2023, 1, 1, 12, 0, 0),
        updated_at=datetime(2023, 1, 2, 12, 0, 0)
    )
    
    # Test model_dump_json
    try:
        json_str = think_entity.model_dump_json()
        data = json.loads(json_str)
        print(f"  model_dump_json(): SUCCESS")
        print(f"  created_at: {data['created_at']}")
        print(f"  updated_at: {data['updated_at']}")
    except Exception as e:
        print(f"  model_dump_json(): FAILED - {e}")
    
    # Test model_dump + json.dumps
    try:
        data = think_entity.model_dump()
        json_str = json.dumps(data)
        print(f"  model_dump() + json.dumps(): SUCCESS")
    except Exception as e:
        print(f"  model_dump() + json.dumps(): FAILED - {e}")

if __name__ == "__main__":
    test_entity_serialization()
    test_relation_serialization()
    test_journal_record_serialization()
    test_think_entity_serialization()
    print("\nAll tests complete!") 