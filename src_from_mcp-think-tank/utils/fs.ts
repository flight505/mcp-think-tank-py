import * as fs from 'fs';
import * as path from 'path';

/**
 * Creates a directory and any parent directories if they don't exist
 * @param dirPath - The path to create
 */
export function createDirectory(dirPath: string): void {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`Created directory: ${dirPath}`);
  }
} 