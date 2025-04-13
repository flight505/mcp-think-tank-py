#!/usr/bin/env python3
"""
Unit tests for the Entity model
"""
import unittest
from datetime import datetime
from src.tools.memory import Entity

class TestEntity(unittest.TestCase):
    """Test suite for the Entity class."""
    
    def test_entity_initialization(self):
        """Test that the Entity model initializes correctly."""
        # Create an entity
        entity = Entity(name="test-entity", entityType="test")
        
        # Verify basic properties
        self.assertEqual(entity.name, "test-entity")
        self.assertEqual(entity.entity_type, "test")
        self.assertIsInstance(entity.observations, list)
        self.assertEqual(len(entity.observations), 0)
        self.assertFalse(entity.deleted)
        self.assertIsNone(entity.embedding)
        self.assertIsInstance(entity.created_at, datetime)
        self.assertIsNone(entity.updated_at)
    
    def test_mutable_default_value(self):
        """Test that the observations field is not shared between instances."""
        # Create two entities
        entity1 = Entity(name="entity1", entityType="test")
        entity2 = Entity(name="entity2", entityType="test")
        
        # Add an observation to entity1
        entity1.observations.append("Test observation for entity1")
        
        # Verify entity2's observations are still empty
        self.assertEqual(len(entity2.observations), 0)
        
    def test_alias_field(self):
        """Test that the alias field works correctly."""
        # Create an entity using the alias
        entity = Entity(name="test-entity", entityType="test-type")
        
        # Verify the field is set correctly
        self.assertEqual(entity.entity_type, "test-type")

if __name__ == '__main__':
    unittest.main() 