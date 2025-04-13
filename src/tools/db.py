#!/usr/bin/env python3
"""
Database utilities for MCP Think Tank
Provides SQLite database connections and helper functions
"""
import os
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Generator, Callable, TypeVar, Generic

logger = logging.getLogger("mcp-think-tank.db")

T = TypeVar('T')

class Database:
    """SQLite database connection manager"""

    def __init__(self, db_path: str):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._ensure_path()
        
    def _ensure_path(self) -> None:
        """Ensure the directory for the database file exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
    def connect(self) -> sqlite3.Connection:
        """
        Get a database connection, creating one if needed

        Returns:
            SQLite connection object
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Return rows as dictionaries
            self.conn.row_factory = sqlite3.Row
            
        return self.conn
        
    def close(self) -> None:
        """Close the database connection if open"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            
    def execute(
        self, 
        query: str, 
        params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> sqlite3.Cursor:
        """
        Execute a SQL query with parameters

        Args:
            query: SQL query
            params: Query parameters (tuple or dict)

        Returns:
            Query cursor
        """
        conn = self.connect()
        try:
            if params is None:
                return conn.execute(query)
            else:
                return conn.execute(query, params)
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}, Query: {query}, Params: {params}")
            raise
            
    def executemany(
        self, 
        query: str, 
        params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]
    ) -> sqlite3.Cursor:
        """
        Execute a SQL query with multiple parameter sets

        Args:
            query: SQL query
            params_list: List of parameter sets

        Returns:
            Query cursor
        """
        conn = self.connect()
        try:
            return conn.executemany(query, params_list)
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}, Query: {query}, Params: {params_list}")
            raise
            
    def query(
        self, 
        query: str, 
        params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return all results as dictionaries

        Args:
            query: SQL query
            params: Query parameters (tuple or dict)

        Returns:
            List of dictionaries representing rows
        """
        cursor = self.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    def query_one(
        self, 
        query: str, 
        params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a SQL query and return the first result as dictionary

        Args:
            query: SQL query
            params: Query parameters (tuple or dict)

        Returns:
            Dictionary representing first row or None if no results
        """
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None
        
    def transaction(self) -> 'Transaction':
        """
        Start a database transaction

        Returns:
            Transaction object
        """
        return Transaction(self)
        
    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """
        Create a table if it doesn't exist

        Args:
            table_name: Name of the table
            schema: Dictionary of column definitions {column_name: type_definition}
        """
        columns = ", ".join(f"{name} {definition}" for name, definition in schema.items())
        self.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
        self.commit()
        
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists

        Args:
            table_name: Name of the table

        Returns:
            True if table exists, False otherwise
        """
        result = self.query_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return result is not None
        
    def commit(self) -> None:
        """Commit current transaction"""
        if self.conn is not None:
            self.conn.commit()
            
    def rollback(self) -> None:
        """Rollback current transaction"""
        if self.conn is not None:
            self.conn.rollback()
            
    def __enter__(self) -> 'Database':
        """Context manager entry"""
        self.connect()
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.close()


class Transaction:
    """Database transaction context manager"""
    
    def __init__(self, database: Database):
        """
        Initialize transaction

        Args:
            database: Database instance
        """
        self.database = database
        self.conn = None
        
    def __enter__(self) -> sqlite3.Connection:
        """Begin transaction and return connection"""
        self.conn = self.database.connect()
        return self.conn
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        End transaction with commit or rollback

        Commits if no exception occurred, rolls back otherwise
        """
        if self.conn is not None:
            if exc_type is not None:
                self.conn.rollback()
            else:
                self.conn.commit()


class Repository(Generic[T]):
    """Base repository class for database operations"""
    
    def __init__(self, database: Database, table_name: str):
        """
        Initialize repository

        Args:
            database: Database instance
            table_name: Name of the table
        """
        self.database = database
        self.table_name = table_name
        
    def create_table_if_not_exists(self, schema: Dict[str, str]) -> None:
        """
        Create the table if it doesn't exist

        Args:
            schema: Dictionary of column definitions {column_name: type_definition}
        """
        self.database.create_table(self.table_name, schema)
        
    def insert(self, data: Dict[str, Any]) -> int:
        """
        Insert a record into the table

        Args:
            data: Dictionary of column values

        Returns:
            ID of the inserted record
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(":" + key for key in data.keys())
        
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        cursor = self.database.execute(query, data)
        self.database.commit()
        
        return cursor.lastrowid or 0
        
    def insert_many(self, data_list: List[Dict[str, Any]]) -> None:
        """
        Insert multiple records into the table

        Args:
            data_list: List of dictionaries with column values
        """
        if not data_list:
            return
            
        # All dictionaries should have the same keys
        columns = data_list[0].keys()
        columns_str = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        
        query = f"INSERT INTO {self.table_name} ({columns_str}) VALUES ({placeholders})"
        
        # Convert list of dicts to list of tuples
        params_list = [tuple(item[col] for col in columns) for item in data_list]
        
        self.database.executemany(query, params_list)
        self.database.commit()
        
    def update(self, id_column: str, id_value: Any, data: Dict[str, Any]) -> None:
        """
        Update a record in the table

        Args:
            id_column: Name of the ID column
            id_value: Value of the ID
            data: Dictionary of column values to update
        """
        set_clause = ", ".join(f"{key} = :{key}" for key in data.keys())
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE {id_column} = :id"
        
        params = {**data, "id": id_value}
        self.database.execute(query, params)
        self.database.commit()
        
    def delete(self, id_column: str, id_value: Any) -> None:
        """
        Delete a record from the table

        Args:
            id_column: Name of the ID column
            id_value: Value of the ID
        """
        query = f"DELETE FROM {self.table_name} WHERE {id_column} = ?"
        self.database.execute(query, (id_value,))
        self.database.commit()
        
    def find_by_id(self, id_column: str, id_value: Any) -> Optional[Dict[str, Any]]:
        """
        Find a record by ID

        Args:
            id_column: Name of the ID column
            id_value: Value of the ID

        Returns:
            Record as dictionary or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE {id_column} = ?"
        return self.database.query_one(query, (id_value,))
        
    def find_all(self) -> List[Dict[str, Any]]:
        """
        Find all records in the table

        Returns:
            List of records as dictionaries
        """
        query = f"SELECT * FROM {self.table_name}"
        return self.database.query(query)
        
    def count(self) -> int:
        """
        Count records in the table

        Returns:
            Number of records
        """
        query = f"SELECT COUNT(*) as count FROM {self.table_name}"
        result = self.database.query_one(query)
        return result["count"] if result else 0 