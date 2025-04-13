import { z } from 'zod';

// Entity validation schema
const EntitySchema = z.object({
  name: z.string().min(1).describe('Unique identifier for the entity'),
  entityType: z.string().min(1).describe('Type classification of the entity'),
  observations: z.array(z.string()).describe('Facts or observations about the entity')
});

// Relation validation schema
const RelationSchema = z.object({
  from: z.string().min(1).describe('Source entity name'),
  to: z.string().min(1).describe('Target entity name'),
  relationType: z.string().min(1).describe('Type of relationship (in active voice)')
});

// Create entities schema
export const CreateEntitiesSchema = z.object({
  entities: z.array(EntitySchema).describe('Array of entities to create')
});

// Create relations schema
export const CreateRelationsSchema = z.object({
  relations: z.array(RelationSchema).describe('Array of relations to create')
});

// Add observations schema
export const AddObservationsSchema = z.object({
  observations: z.array(z.object({
    entityName: z.string().min(1).describe('Name of the entity to add observations to'),
    contents: z.array(z.string()).describe('Observations to add to the entity')
  })).describe('Array of entity observations to add')
});

// Delete entities schema
export const DeleteEntitiesSchema = z.object({
  entityNames: z.array(z.string().min(1)).describe('Array of entity names to delete')
});

// Delete observations schema
export const DeleteObservationsSchema = z.object({
  deletions: z.array(z.object({
    entityName: z.string().min(1).describe('Name of the entity to remove observations from'),
    observations: z.array(z.string()).describe('Observations to remove from the entity')
  })).describe('Array of entity observations to delete')
});

// Delete relations schema
export const DeleteRelationsSchema = z.object({
  relations: z.array(RelationSchema).describe('Array of relations to delete')
});

// Search nodes schema
export const SearchNodesSchema = z.object({
  query: z.string().min(1).describe('Search query to find matching entities')
});

// Open nodes schema
export const OpenNodesSchema = z.object({
  names: z.array(z.string().min(1)).describe('Array of entity names to retrieve')
});

// Update entities schema
export const UpdateEntitiesSchema = z.object({
  entities: z.array(z.object({
    name: z.string().min(1).describe('Name of the entity to update'),
    entityType: z.string().optional().describe('New entity type (optional)'),
    observations: z.array(z.string()).optional().describe('New observations to replace existing ones (optional)')
  })).describe('Array of entities to update')
});

// Update relations schema
export const UpdateRelationsSchema = z.object({
  relations: z.array(RelationSchema).describe('Array of relations to update')
}); 