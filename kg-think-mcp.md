# Implementation for MCP Think Tank

# MCP Think Tank (Python)

This is the Python implementation of the MCP Think Tank using FastMCP V2.

> **Note**: This is a complete rewrite of the [original TypeScript version](https://github.com/flight505/mcp-think-tank) 
> to take advantage of FastMCP V2's improved timeout handling and additional features.
This new version of mcp-think-tank will be on https://github.com/flight505/mcp-think-tank-py
Just for reference the old src is avalible here: src_from_mcp-think-tank in the root folder. 

links that might be usefull:
https://github.com/jlowin/fastmcp
https://github.com/modelcontextprotocol/python-sdk
https://modelcontextprotocol.io/tutorials/building-mcp-with-llms
https://modelcontextprotocol.io/docs/concepts/architecture
https://www.anthropic.com/news/visible-extended-thinking
https://github.com/PhillipRt/think-mcp-server
https://github.com/itseasy21/mcp-knowledge-graph
https://github.com/eyaltoledano/claude-task-master
https://simonwillison.net/2025/Mar/21/the-think-tool/#:~:text=%7B%20,



## Key Differences from TypeScript Version

- **Installation**: Uses Python package managers (uv) instead of npm
- **Configuration**: Uses FastMCP's built-in Claude Desktop integration and needs to work with both Cursor and Claude Desktop
- **Performance**: Better timeout handling - timer resets after each tool call
- **Features**: Additional capabilities like context injection and server composition

# MCP Think Tank – Next-Generation Implementation Plan (Pushing the Frontier)

Below is a **comprehensive** and **detailed** roadmap for creating a next-generation MCP Think Tank server in Python. Unlike earlier outlines, this plan **avoids "optional" disclaimers** and **commits** to the most advanced features and best practices to produce a **significant improvement**—one that developers will truly **want** to use. It assumes:

- **We use `uv`** for the Python environment and concurrency.
- **We store the knowledge graph in `.jsonl`** files for incremental append and partial updates.
- **We want synergy between** memory (KG), advanced reasoning (think tool), and a robust task management system.
- **We want advanced orchestration, embedding-based search,** and other "frontier" features integrated in a single server.

## Phase 1: Repository & System Initialization

- [x] **Repository Setup & Tools**
  - [x] Create or confirm a dedicated repo, e.g. `mcp-think-tank-nextgen/`.
  - [x] Initialize a Python environment with `uv`:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install fastmcp pydantic ...
    ```
  - [x] Generate a `pyproject.toml`
   listing core dependencies:
    - `fastmcp >= 2.0.0`
    - `pydantic >= 2.0.0`
    - `tqdm` (for progress feedback), 
    - `sentence-transformers` or `langchain` for embeddings (detailed below).
  - [x] Prepare a concise but **powerful** README that sets expectations about the advanced features.

- [x] **Directory Layout (Emphasizing Modularity + Orchestration)**

- **Installation Method**: Python package managers (uv)
- [ ] **Distribution**: New package distribution through PyPI, once we er compleatly done we have to research how to best destirbut the mcp server, for many users installing via npx is prefered, we might need to look into a wrapper or some better options. 
- **Architecture**: Leveraging FastMCP V2's decorator-based approach and advanced features

mcp-think-tank-nextgen/
├── src/
│   ├── server.py             # Main server entry point
│   ├── config.py             # Central config (ENV variables, constants)
│   ├── orchestrator.py       # Orchestration layer for multi-tool synergy
│   ├── tools/
│   │   ├── think.py          # Advanced structured reasoning tool
│   │   ├── memory.py         # Knowledge graph + memory handling
│   │   ├── tasks.py          # Task management, multi-step planning
│   │   ├── embeddings.py     # Vector-based retrieval (semantic memory)
│   │   └── init.py
│   ├── watchers/
│   │   └── file_watcher.py   # (If we do codebase indexing, watchers, etc.)
│   └── …
├── tests/
├── .venv/
├── README.md
└── pyproject.toml

- [x] **Core Concurrency & Orchestrator**
- [x] Implement `orchestrator.py` to manage cross-tool intelligence. This is **not** optional. It ensures:
  - Automatic retrieval of relevant memory context when the user or Claude references certain entities/keywords.
  - Coordinated calls to the task tool after memory or think steps. 
  - A "reflection loop" if needed (detailed in the reasoning phase).

- [x] **Config & Env Variables**
- [x] Centralize environment variable handling in `config.py`:
  - `MEMORY_FILE_PATH` defaulting to `~/.mcp-think-tank/memory.jsonl`.
  - `USE_EMBEDDINGS` (Boolean) to toggle advanced embedding retrieval. If unset, defaults to True (since we are pushing the frontier).
  - `ANTHROPIC_API_KEY` or any other necessary keys for advanced calls.
- [x] Provide robust error messages if keys/paths are missing, since we aim for a production-grade solution.

---

## Phase 2: Advanced Knowledge Graph with `.jsonl` Storage

- [x] **Incremental JSONL Persistence**
- [x] **Design the `.jsonl` format** so each line is either:
  - A record describing an **entity creation** (with `name`, `type`, `observations`).
  - A record describing a **relation** (with `from`, `to`, `relation_type`).
  - A record describing **update** or **deletion**.
- [x] On server startup, parse every line in `.jsonl` to rebuild the in-memory data structures:
  - `entities: Dict[str, Entity]`
  - `relations: List[Relation]` or a more advanced adjacency structure.
- [x] Each time we add or modify an entity/relation, **append** a new line reflecting the change. 
  - For deletes, append a line marking the item as deleted. 
  - Regularly (e.g. weekly) auto-compact the file to remove old or overwritten lines.

- [x] **KnowledgeGraph Class (Memory)**
- [x] Provide methods:
  - `add_entity(...)`: checks memory first; if new, adds in memory & appends to `.jsonl`.
  - `add_relation(...)`: same approach, ensures no duplicates. 
  - `update_entity(...)`: if we want to add new observations or rename an entity.
  - `delete_entity(...)`: appends a delete marker, removes from memory dict.
  - `search_nodes(query: str, use_semantic: bool = True)`: by default, do **semantic** search (detailed below).
- [x] Include metadata (timestamps, user/system actor) to track who/what caused each line.
- [x] Maintain an in-memory index for immediate lookups; `.jsonl` is the single source of truth on disk.

- [x] **Automated Embeddings & Semantic Search** (**Mandatory**)
- [x] Use `sentence-transformers` or `langchain` to generate embeddings for each entity's observations.
- [x] Store these embeddings in memory (e.g. in a `Dict[str, np.ndarray]`) or optionally in `.jsonl` for fallback re-load.
- [x] `search_nodes(query, use_semantic=True)`:
  - Convert `query` → embedding, compute similarity with each entity's vector, and return top-k matches.
  - Fall back on keyword matching if no embeddings are found or if `USE_EMBEDDINGS=False`.
- [x] On each new observation, auto-generate embeddings and store them. Re-append to `.jsonl` with the vector data or store offline in a parallel file.

- [x] **Error Handling & Logging**
- [x] Any file I/O error (permissions, disk full) should be logged with a robust message and not crash the server.
- [x] Maintain a minimal local log (e.g. `mcp-think-tank.log`) for diagnosing memory changes or failures.

---

## Phase 3: "Think" Tool – Structured Reasoning & Reflexion

- [x] **Core Think Functionality**
- [x] Implement `think(structured_reasoning: str, store_in_memory: bool = False, reflexion: bool = True) -> str`:
  - Returns or logs the structured reasoning.
  - If `store_in_memory=True`, create an entity with type `Thought` and store the user's reasoning text (plus timestamp).
  - If `reflexion=True`, triggers a **reflection pass** (see below).

- [x] **Reflection & Self-Evaluation Pass**
- [x] Inside the think tool, after storing the user reasoning, optionally call a second step:
  - Provide the user's original reasoning plus a "self-check prompt" to see if any immediate corrections are needed.
  - Let the tool respond with a "Refined Thought" or confirm the original is correct.
  - Store that refined thought in memory as well if it's different from the original.
- [x] This enables advanced chain-of-thought refinement, pushing the frontier for multi-step self-correction.

- [x] **Context Injection from Memory**
- [x] If `query` or `topic` is given to `think`, automatically do a semantic search (`search_nodes(query)`) and inject the top 2–3 relevant memory items for the user's structured reasoning. 
- [x] Append a "context" block to the final output: "Context from Memory: ...".
- [x] This ensures the user's reasoning is well-informed by existing knowledge graph entries.

- [x] **Interface/Docstrings**
- [x] Provide thorough docstrings so that LLMs (Claude) know exactly how to use the tool for reflection, with examples:
  - "Call `think` before proceeding to the next action if you need to reflect on the user request or memory context."

---

## Phase 4: Task Management & Multi-Step Planning

- [x] **Advanced Task Tool**
- [x] Implement `create_tasks(prd_text: str) -> str`:
  - Takes a project requirement or prompt, calls out to an internal function that uses **Anthropic** (Claude) or local LLM to parse tasks.
  - Stores each new task as a distinct "Task" entity in the knowledge graph, with fields like `title`, `status`, `priority`.
- [x] Implement `list_tasks() -> str`:
  - Retrieves all "Task" entities from memory, returns them in a structured format (or JSON).
- [x] Implement `update_task(task_id: str, updates: dict) -> str`:
  - Merges new data (e.g. `status=done`, `priority=high`) into the task entity.
  - Appends the update in `.jsonl`.
- [x] Implement `delete_task(task_id: str) -> str`:
  - Marks the "Task" entity as deleted in memory, appends a deletion record.

- [x] **Orchestrator-Driven Planning**
- [x] Within `orchestrator.py`, define a routine that automatically:
  - Calls `create_tasks` if the user says "Plan my project" or references a PRD.
  - As tasks are created, calls the think tool for a quick reflection pass (e.g. "Do these tasks make sense? Are they complete?").
  - Potentially queries memory for relevant domain knowledge to refine tasks.
- [x] Provide an orchestrator function `auto_plan_workflow` that can chain:
  1. Parse PRD → `create_tasks`
  2. `think` about tasks → reflection
  3. `update_task` if reflection suggests changes
  4. Store final tasks in memory with semantic embeddings
  - Return a final summary to the user.

- [x] **LLM Integration for Task Parsing** *(Implementation Complete)*
- [x] Implement **Option 1: Initial Setup Download** approach:
  - Primary parser uses **Anthropic API** when available
  - Fallback parser uses **Gemma 3 1B** model (~900MB) via Hugging Face
  - Includes model download option during initial MCP setup/installation
  - Creates wrapper function to select appropriate parser based on availability
- [x] Handle concurrency with `asyncio`, including error handling and timeouts.
- [x] Show progress information during model download and setup.

---

## Phase 5: File System Awareness & Codebase Integration

- [ ] **File Watchers** (Fully Integrated, Not Optional)
- [ ] Implement `file_watcher.py` that:
  - Recursively indexes `.py`, `.md`, `.js`, etc. in the user's codebase (based on an env variable like `PROJECT_PATH`).
  - Maintains a map of file → summary or key definitions (classes, functions).
  - On changes, re-index and update a "File" or "Code" entity in the knowledge graph with relevant observations.

- [ ] **Search Code / Summarize Code Tools**
- [ ] Add `search_code(query: str) -> str`:
  - Internally queries the watchers' index. Optionally combine it with the semantic search approach if we embed code or docstrings.
  - Returns snippet references and line numbers for relevant matches.
- [ ] Add `summarize_file(filepath: str) -> str`:
  - Possibly calls an LLM to produce a summarized overview of the code or doc text.
  - Stores that summary in the knowledge graph for future reference.

- [ ] **Auto-Context Injection from Code to Memory**
- [ ] Whenever user or Claude references a function or file, orchestrator checks watchers' index:
  - If relevant, inject a snippet or summary into the conversation.
  - Allow the user to `think` on it or store an observation in memory.

---

## Phase 6: Multi-Tool Orchestration & Performance

- [ ] **Global Orchestrator with Directed Graph of Steps**
- [ ] In `orchestrator.py`, define a data flow for typical advanced tasks:
  - Example: "Implement Feature X" → (1) parse PRD → (2) create tasks → (3) search code → (4) think and reflect → (5) finalize tasks, store in memory.
- [ ] Provide an API so that Claude (via the conversation) can call `mcp.run_workflow("build_feature")`, which triggers the orchestrated steps behind the scenes. Return intermediate states to Claude.

- [ ] **Timeout and Error Recovery**
- [ ] For each tool call, ensure we have a well-defined timeout (e.g. 30s) using `asyncio.wait_for` or built-in FastMCP timeouts.
- [ ] If a step fails (network error, disk error), the orchestrator tries a fallback path or logs the failure for the user. 
- [ ] No single error should crash the entire server; the system must degrade gracefully.

- [ ] **Embedding Cache & Performance**
- [ ] Since we generate embeddings for each new entity/observation, store them in memory to avoid repeated re-encodings. 
- [ ] For large `.jsonl` knowledge bases, implement a partial load or indexing strategy to keep search within feasible latency.

- [ ] **Logging & Metrics**
- [ ] All major tool calls should log usage in `mcp-think-tank.log`, including timestamps, parameters, and success/failure. 
- [ ] (Potentially) implement usage analytics to measure how often each tool is called, average search time, etc., to keep performance in check.

---

## Phase 7: Rigorous Testing & Documentation

- [ ] **Unit Tests (Mandatory)**
- [ ] Memory Tests:
  - Add, update, delete entities, relations. Check `.jsonl` lines appended. Verify search by keyword & embedding.
- [ ] Think Tool Tests:
  - Provide short reasoning strings, check reflection output. 
  - Provide a big chunk of text, ensure no crashing or timeout.
- [ ] Task Tool Tests:
  - Create tasks from sample PRD, list them, update statuses. Confirm `.jsonl` changes. 
  - Trigger orchestrator workflows in a test scenario to confirm correct chaining.
- [ ] File Watcher Tests:
  - Mock or create a small codebase, run watchers, ensure code references are stored. 
  - Edit a file, confirm the knowledge graph updates.

- [ ] **Integration & E2E Tests**
- [ ] Simulate a complex user request that triggers a chain: "Plan my new microservice." 
  - (1) "Think" about it, (2) create tasks from PRD, (3) store tasks in memory, (4) reflect on next steps, (5) finalize plan. 
  - Verify all steps succeed and `.jsonl` gets consistent records.
- [ ] Put the system under moderate load (multiple requests in quick succession) to confirm concurrency handling.

- [ ] **Comprehensive Documentation**
- [ ] README / Wiki describing each advanced feature:
  - Orchestrator usage, `.jsonl` format, reflection mechanism, watchers, code search, etc.
- [ ] Clear instructions for installing with `uv` and integrating with both Cursor & Claude Desktop. 
- [ ] Example dialogues showing how each tool might be invoked by Claude, ensuring discoverability.
- [ ] Examples should be foused on devellopment (coding) 
- [ ] README should look very cool with badges and all the newest github details such as https://www.star-history.com/#eyaltoledano/claude-task-master&Timeline . We have images for logo in light and dark mode assets/MCP_Think_Tank_light.png & assets/MCP_Think_Tank_dark.png , size should be around 300. We also have Favicon's ready.  
- [ ] The better and more detailed the README is the more users we will get, so focus on creating very well documented README with examples, and accurate instalation detailes for Cursor and Claude and other programs that can use our MCP server. 

---

## Phase 8: Deployment, Distribution & Community Adoption

- [ ] **Single-Command Installation**
- [ ] Provide a script or instructions: 
  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install .
  fastmcp install src/server.py
  ```
  That sets up everything for local usage, whether on Cursor or Claude Desktop.

- [ ] **Automatic Registration with Claude Desktop**
- [ ] Implement a `fastmcp install` target or a Smithery script that inserts a config entry for Claude Desktop usage. 
- [ ] Validate that after installation, the tools appear in Claude Desktop's UI.

- [ ] **Marketing & Community Docs**
- [ ] Emphasize the **frontier** features: reflection, orchestrator-based synergy, embedding-based knowledge graph, etc.
- [ ] Provide a compelling use-case scenario so developers see the advantage over simpler or "optional" solutions.

- [ ] **Continuous Improvement**
- [ ] Listen to user feedback for performance or memory usage issues. 
- [ ] Keep the code robust, push further with new LLM integration or vector indexing if beneficial.
- [ ] Expand watchers for more file types, advanced doc parsing, or more refined code summarization.
- [ ] At this point we need to revisit the README make sure should we update the distribution detailes. Again the better and more detailed the README is the more users we will get, so focus on creating very well documented README with examples, and accurate instalation detailes for Cursor and Claude and other programs that can use our MCP server. Many users is the goal!! 
- [ ] Write a cool website that showcases the MCP server "MCP Think Tank" images and favicons are in public folder. We will publish the website on my Vercel account. and link it to the github repo. I will not buy a domain name specifically for this project. 

---

## Final Notes

By **combining** advanced KG storage (incremental `.jsonl`, embedding-based search), a **reflective think tool**, robust **task management** with orchestrated multi-step workflows, **file watchers**, and thorough concurrency/error handling, this MCP Think Tank goes well beyond basic feature sets. We **commit** to these advanced features—none are relegated to optional or "maybe later" status.

The net result is an **intelligent, forward-looking** MCP environment that consistently surpasses older servers in reliability, synergy between tools, and real developer productivity. This approach ensures that the entire system, from knowledge graph to orchestrated planning, is integrated into a single codebase that truly pushes the frontier and makes the Think Tank indispensable for AI-driven workflows.

## Future Enhancements

- [ ] Add visualization tool for the knowledge graph