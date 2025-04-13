import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { KnowledgeGraph } from './knowledgeGraph.js';

/**
 * Class responsible for persisting the knowledge graph to disk
 */
export class GraphStorage {
  private filePath: string;
  private graph: KnowledgeGraph;

  /**
   * Create a new GraphStorage instance
   * @param filePath - Path to the storage file
   * @param graph - Knowledge graph to persist
   */
  constructor(filePath: string, graph: KnowledgeGraph) {
    this.filePath = filePath;
    this.graph = graph;
    this.load();
  }

  /**
   * Load the graph from the file
   */
  load(): void {
    try {
      if (fs.existsSync(this.filePath)) {
        const data = fs.readFileSync(this.filePath, 'utf8');
        const jsonData = JSON.parse(data);
        this.graph.fromJSON(jsonData);
        console.log(`Loaded graph from ${this.filePath}`);
      } else {
        console.log(`No existing graph found at ${this.filePath}, starting with empty graph`);
      }
    } catch (error) {
      console.error(`Error loading graph: ${error}`);
    }
  }

  /**
   * Save the graph to the file
   */
  save(): void {
    try {
      const data = JSON.stringify(this.graph.toJSON(), null, 2);
      fs.writeFileSync(this.filePath, data, 'utf8');
      console.log(`Saved graph to ${this.filePath}`);
    } catch (error) {
      console.error(`Error saving graph: ${error}`);
    }
  }

  /**
   * Log an operation for debugging
   * @param operation - The operation being performed
   * @param details - Details about the operation
   */
  logOperation(operation: string, details: any): void {
    console.log(`[${operation}] ${JSON.stringify(details)}`);
  }
}

// Create a default graph instance
const memoryPath = process.env.MEMORY_PATH || path.join(os.homedir(), '.mcp-think-tank/memory.jsonl');

// Export the graph and storage for use in tools
export const graph = new KnowledgeGraph();
export const graphStorage = new GraphStorage(memoryPath, graph);

// We'll look at the file first to understand its implementation 