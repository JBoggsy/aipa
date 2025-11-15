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
* There are definitely big questions about how to handle different agents for different tasks.
  Should agents be task-specialized? It certainly seems so, as that's the primary function of
  separating agents out. However, it might make more sense to have a single task-completeing agent
  with different tasks. Worth experimentnig with.
* It seems like cycling and pro-activity might not be the most effective approach out of the gate.
  I've been vacillating on this question for too long. In the name water flowing through simple
  pipes, I think I'll move to a coordinated model for now. 

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

## Tools
* The models I've experimented with (smolLMv3, gpt-oss:20b) are usually pretty good about calling
  tools correctly and not hallucinating tools *as long as they are given the tools they expect*. 
* One issue is creating tools from agents. The tool-agents cannot have themselves as tools or they
  will just attempt to call themselves
* It is frustratingly hacky to try and use instance or class methods as tools because the agent
  cannot provide the `self` or `cls` argument required, so you need to create a factory pattern
  which generates a non-`self` method, but you cannot just use a normal Python decorator because the
  generated method needs accurate docstrings. The tension is that without the `self` argument, the
  tool doesn't have access to the `Agent` instance's various attributes. I'm considering moving
  user/agent context to the global namespace so that tool functions can always access that
  information.
* The previous point is only an issue because I'm using the function-to-tool conversion in Ollama.
  If I could figure out how to properly define tool JSON schemas, I could use `self` methods.
* Another alternative: only have tools which don't require user/agent context.
* I've ended up creating a `Tool` base class, yet another reinvention of the wheel.

## User and Agent Context
* One big question I've run into: how do we maintain user and agent context and make it accessible
  to (and updateable by) the agent. There are lots of different types of contextual information,
  each with different needs in how they're handled by the agent, how long they need to remain in the
  agent's "working memory," and so on.
* The concept of the agent's "working memory" is still something I'm working out. How is it added
  to, used, manipulated, etc? How do we get the LLM to actually reason properly about what it knows
  and what it should do next? 
* As an example, how should the agent handle user emails? How do it parse them and extract both high-
  and low-priority action items, information about the user, opportunities, etc. Ads, for example,
  are very low prio *except* that sometimes the user might actually be interested in the product. 
* Another consideration is allowing tools/agents to access user/agent context. One solution to the
  problem of needing tool factory methods (see Tools above) is to move requisite contextual
  information to a global namespace. This might work for the user context, but each agent has its
  own context and they can't all be in the global namespace.


## Lessons re: continuous cycling
By "continuous cycling" I mean have the agent prompting itself for its next steps on a cycle,
cognitive architecture style. The approach I took was to have a `cycle_step` method that either
prompted the agent to come up with tasks for itself if it had none or select and achieve a task if
it did.The effectiveness of this was very dependent on prompting instructions and on tools. Given a
clear context statement and a tool which clearly matches that context statement it generally did
well. However, the guidance of the context statements only went so far. In a task with a few
sub-tasks, the agent would occasionally just do a sub-task and then get confused. Additionally, the
agent gets lost fairly easily without enough prompt guidance. It chooses things which are reasonable
but aren't effective. Thus, two components are particularly crucial: prompts which clearly guide the
agent and tools which are directly related to the agent's tasks. Given these two things, the agent
can consistently do a pretty good job.

Also, keeping the temperature low seems to be more effective.