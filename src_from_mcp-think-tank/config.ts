import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';
import minimist from 'minimist';

// Parse command line arguments
const argv = minimist(process.argv.slice(2));

// Default memory path
const DEFAULT_MEMORY_PATH = join(homedir(), '.mcp-think-tank', 'memory');

// Create default directories if they don't exist
if (!existsSync(DEFAULT_MEMORY_PATH)) {
  mkdirSync(DEFAULT_MEMORY_PATH, { recursive: true });
}

// Export configuration object
export const config = {
  // Memory path from command line or default
  memoryPath: argv['memory-path'] || DEFAULT_MEMORY_PATH,
  
  // Request timeout in milliseconds (default: 300 seconds = 5 minutes)
  requestTimeout: argv['request-timeout'] ? 
                 parseInt(argv['request-timeout'] as string, 10) * 1000 : 
                 process.env.REQUEST_TIMEOUT ? 
                 parseInt(process.env.REQUEST_TIMEOUT, 10) * 1000 : 
                 300000,
  
  // Other config options
  debug: !!argv.debug,
  version: '1.0.5', // Should match package.json
};

export default config; 