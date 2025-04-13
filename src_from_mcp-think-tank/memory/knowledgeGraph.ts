/**
 * Entity interface
 */
export interface Entity {
  name: string;
  entityType: string;
  observations: string[];
}

/**
 * Relation interface 
 */
export interface Relation {
  from: string;
  to: string;
  relationType: string;
}

/**
 * KnowledgeGraph class for managing entities and relations
 */
export class KnowledgeGraph {
  entities: Map<string, Entity>;
  relations: Map<string, Set<Relation>>;

  constructor() {
    this.entities = new Map();
    this.relations = new Map();
  }

  /**
   * Add a new entity to the graph
   * @param entity - The entity to add
   * @returns true if added, false if already exists
   */
  addEntity(entity: Entity): boolean {
    if (this.entities.has(entity.name)) {
      return false;
    }
    
    // Create a copy of the entity with all properties
    const newEntity: Entity = {
      name: entity.name,
      entityType: entity.entityType,
      observations: [...entity.observations]
    };
    
    this.entities.set(entity.name, newEntity);
    
    return true;
  }

  /**
   * Add a new relation between entities
   * @param relation - The relation to add
   * @returns true if added, false if invalid or already exists
   */
  addRelation(relation: Relation): boolean {
    // Check if both entities exist
    if (!this.entities.has(relation.from) || !this.entities.has(relation.to)) {
      return false;
    }
    
    // Create relation set for source entity if it doesn't exist
    if (!this.relations.has(relation.from)) {
      this.relations.set(relation.from, new Set());
    }
    
    // Check if relation already exists
    const relations = this.relations.get(relation.from);
    const existingRelation = Array.from(relations!).find(r => 
      r.from === relation.from && 
      r.to === relation.to && 
      r.relationType === relation.relationType
    );
    
    if (existingRelation) {
      return false;
    }
    
    // Add the relation
    relations!.add(relation);
    return true;
  }

  /**
   * Add observations to an entity
   * @param entityName - The name of the entity
   * @param observations - The observations to add
   * @returns Array of added observations
   */
  addObservations(entityName: string, observations: string[]): string[] {
    const entity = this.entities.get(entityName);
    if (!entity) {
      return [];
    }
    
    const added: string[] = [];
    for (const observation of observations) {
      if (!entity.observations.includes(observation)) {
        entity.observations.push(observation);
        added.push(observation);
      }
    }
    
    return added;
  }

  /**
   * Delete observations from an entity
   * @param entityName - The name of the entity
   * @param observations - The observations to delete
   * @returns true if entity exists, false otherwise
   */
  deleteObservations(entityName: string, observations: string[]): boolean {
    const entity = this.entities.get(entityName);
    if (!entity) {
      return false;
    }
    
    // Remove each observation if it exists
    for (const observation of observations) {
      const index = entity.observations.indexOf(observation);
      if (index !== -1) {
        entity.observations.splice(index, 1);
      }
    }
    
    return true;
  }

  /**
   * Delete an entity and its relations
   * @param entityName - The name of the entity to delete
   * @returns true if deleted, false if not found
   */
  deleteEntity(entityName: string): boolean {
    if (!this.entities.has(entityName)) {
      return false;
    }
    
    // Delete the entity
    this.entities.delete(entityName);
    
    // Delete relations where this entity is the source
    this.relations.delete(entityName);
    
    // Delete relations where this entity is the target
    for (const [source, relations] of this.relations.entries()) {
      const newRelations = Array.from(relations).filter(r => r.to !== entityName);
      this.relations.set(source, new Set(newRelations));
    }
    
    return true;
  }

  /**
   * Delete a relation
   * @param relation - The relation to delete
   * @returns true if deleted, false if not found
   */
  deleteRelation(relation: Relation): boolean {
    // Check if source entity exists in relations
    if (!this.relations.has(relation.from)) {
      return false;
    }
    
    const relations = this.relations.get(relation.from);
    
    // Find the relation to delete
    const relationToDelete = Array.from(relations!).find(r => 
      r.from === relation.from && 
      r.to === relation.to && 
      r.relationType === relation.relationType
    );
    
    if (!relationToDelete) {
      return false;
    }
    
    // Delete the relation
    relations!.delete(relationToDelete);
    return true;
  }

  /**
   * Search for entities based on a query
   * @param query - The search query
   * @returns Array of matching entities
   */
  searchNodes(query: string): Entity[] {
    if (!query || query.trim() === '') {
      return [];
    }
    
    const results: Entity[] = [];
    const normalizedQuery = query.toLowerCase();
    
    for (const entity of this.entities.values()) {
      // Check if query matches entity name or type
      if (
        entity.name.toLowerCase().includes(normalizedQuery) ||
        entity.entityType.toLowerCase().includes(normalizedQuery)
      ) {
        results.push(entity);
        continue;
      }
      
      // Check if query matches any observations
      for (const observation of entity.observations) {
        if (observation.toLowerCase().includes(normalizedQuery)) {
          results.push(entity);
          break;
        }
      }
    }
    
    return results;
  }

  /**
   * Get entities by name
   * @param names - Array of entity names
   * @returns Array of found entities
   */
  getEntities(names: string[]): Entity[] {
    const results: Entity[] = [];
    
    for (const name of names) {
      const entity = this.entities.get(name);
      if (entity) {
        results.push(entity);
      }
    }
    
    return results;
  }

  /**
   * Convert the graph to a JSON-serializable object
   * @returns The serialized graph
   */
  toJSON() {
    return {
      entities: Array.from(this.entities.values()),
      relations: Array.from(this.relations.entries()).flatMap(([_, rels]) => Array.from(rels))
    };
  }

  /**
   * Create a knowledge graph from JSON data
   * @param data - The JSON data
   */
  fromJSON(data: any): void {
    // Clear existing data
    this.entities.clear();
    this.relations.clear();
    
    // Add entities
    if (data.entities && Array.isArray(data.entities)) {
      for (const entity of data.entities) {
        this.addEntity(entity);
      }
    }
    
    // Add relations
    if (data.relations && Array.isArray(data.relations)) {
      for (const relation of data.relations) {
        this.addRelation(relation);
      }
    }
  }
} 