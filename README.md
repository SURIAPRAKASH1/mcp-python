# MCP-PYTHON 🔃🔏:

Model Context Protocal Implemenation in Pythonic way. It's small project that embracess MCP. Even most sophisticated model(usually LLM's 🧠) still suffer from not knowing what's going on real-time. Model knowledge is cut-off by some limit. So when an user's query involves questions about real-time data model can't provide reasonable response (Model CoT is like let's Hallucinate 🤪). MCP mitigates this knowledge gap by **Filling context of model by relevant data with user approval**.

# Content 📃:

Am just walking through project's Structure and Explaning what it is ?, How's it used ?, In the universe of MCP. How can be used up to YOU 👊.

## [mcp-client](mcp-client):

### [Host](mcp-client/host.py):

- An application/UI; Anyone (end-user) can interact with. Gradio act as a host. Host is the one contains client & model as in it's top layer.
- Run this command to spin up the host. Change `.../server.py` to match absolute path of mcp server. It will create [gradio] and [client] then [client] will launch [server] as subprocess over **stdio**.

```
uv run mcp-client/host.py --server_command python .../server.py
                      (or)
python mcp-client/host.py --server_command python .../server.py
```

- Run this command to spin up host. It's for client to communicate with streamable-http server. Change `your_mcp_server_url` with actuall url of mcp-server

```
uv run mcp-client/host.py --streamablehttp_url your_mcp_server_url
                      (or)
python mcp-client/host.py --streamablehttp_url your_mcp_server_url

```

[gradio]: /mcp-client/host.py
[client]: /mcp-client/client.py
[server]: /mcp-server/server.py

### [Client](mcp-client/client.py):

- Wraped by host; Initiate communication with server behalf of Host. Provides session/plateform for model to process server Indirectly.
- For e.g, if LLM wants some real data from browser LLM would say here's the function that fetch data from browser hey client i want you do execute this function for me and give back result of this function. Client will take that function to server then server executes function responded with result of that function.

## [mcp-server](mcp-server):

### [server](mcp-server/server.py):

- You can think of server as a piece of code listening for someone to connect and do work for them. We can deploy server as **streamable-http** in a hosting plotform or lauch server as subprocess of client via **stdio**.
- This Implementation provides dynamic way so you can use stdio or streamable-http server based on your personal needs.

- To launch streamable-http server in local run;

```
    python mcp-server/server.py --transport streamable-http
```

### [gradio server](mcp-server/app.py):

- We can launch this server as gradio server. Integrated gradio with mcp-server so we can run this mcp-server as standlone gradio mcp server on huggingface space or launch gradio mcp server in local.

- To lauch gradio mcp server on locally just simply run,

```
python mcp-server/app.py
    (or)
gradio mcp-server/app.py
```

### [Primitives](mcp-server/primitives/):

- Core components of Model Context Protocal.

1.  [tools](mcp-server/primitives/tools.py)
2.  resources
3.  prompt templates
4.  sampling
