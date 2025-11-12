# Learning Notes

This documents records the thoughts, ideas, and realizations I have as I work on this project; and
in particular highlights what I've learned in terms of needs, wants, lessons, etc. when it comes to
building LLM agents and multi-agent systems. I am purposefully using the provider libraries (e.g.,
`openai`, `ollama`, `huggingface` rather than `langchain` or `google-adk`) in order to fully
understand all the different aspects of the task. 

## Agents
* It is helpful, especially when it comes to testing/experimenting, to separate agents from
  underlying LLM models. This allows testing what model and model parameters are most effective for
  the agent. 
* Ideally models should have unified initialization and generation interfaces so that agent code
  doesn't have to care about the underlying model.
* Still need to figure out how I want to handle tools and sub-agents/super-agents, because I don't
  know how best to coordinate/connect them. Currently I envision using sub-agents as a form of tool
  use, where the "tool" in question uses another agent. Some questions/issues with this: who owns
  the sub-agents and how should they be passed to their super-agents? How do we handle using
  multiple models w/ limited compute? Is this actually more efficient than using one agent which has
  many prompts? 
* Agent history, especially tool calling results is probably important, even if the agent isn't
  "conversing" with the user. I need a way to store the historical message logs, but also I don't
  want it to necessarily be "permanent." In other words, there will probably be many "N-step
  reasoning/acting" situations where the agent should be able to accumulate previous responses, tool
  uses, and even user interactions, but I don't want that to be the primary form of memory in the
  system.
* One possible solution: task-level memory which handles tracking the overall object, plan,
  progress, and message history for a specific task.
* I've decided it's better for the agent to handle actually executing tool calls rather than the
  model, since the agent ought to be responsible for tools.

## Models
* Obvious, but using multiple different *local* models takes up a ton of VRAM. This limits using a
  variety of different locally-run models for different agents, though using multiple *hosted*
  models is still feasible.
* I'd really like to be able to use the same model with a variety of different generation parameters
  (temp, top_p/k, reasoning) without needing to re-initialize each time.
* The interfaces from the model libraries are pretty varied, needed to create a unified interface to
  abstract model calling from agent running.

## Messages
* The general approach to messages seems to be a list of dicts with `'role'` and `'content'`
  entries. 
  - Typical roles are `'system'`, `'user'`, and `'assistant'`. Maybe also `'tool'` or similar?
* Need to figure out a unified message format. Different models/providers give essentially the same
  information in different ways. Key pieces of information include the final message content, the
  thinking process, and tool calls. Also the role
* Messages, rather than strings or anything else, should be what model generation returns.
  