import { z } from 'zod';

/**
 * Schema for the think tool parameters
 */
export const ThinkSchema = z.object({
  structuredReasoning: z.string()
    .min(10, 'Reasoning must be at least 10 characters long')
    .describe('A structured thought process to work through complex problems. Use this as a dedicated space for reasoning step-by-step.'),
  
  // Optional memory parameters - can be added in future to associate thoughts with specific contexts
  associateWithEntity: z.string().optional()
    .describe('Optional entity name to associate this thought with'),
  
  category: z.string().optional()
    .describe('Optional category for the thought (e.g., "problem-solving", "analysis", "planning")'),
  
  tags: z.array(z.string()).optional()
    .describe('Optional tags to help categorize and find this thought later'),
  
  storeInMemory: z.boolean().optional()
    .default(false)
    .describe('Whether to store this thought in the knowledge graph memory'),
}); 