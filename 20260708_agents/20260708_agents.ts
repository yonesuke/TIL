import OpenAI from "openai";

// Types for tool call handling
interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

// Tool: say_hello
function sayHello(name: string = ""): string {
  return name ? `Hello ${name}!` : "Hello from OpenRouter!";
}

// Tool: get_weather
function getWeather(location: string): string {
  return `The weather in ${location} is sunny and 22°C.`;
}

// Tool definitions (sent to the API)
const TOOL_DEFINITIONS: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "say_hello",
      description: "Returns a greeting message. Optionally takes a name.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "The name to greet",
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather for a given location.",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "City name, e.g. 'Tokyo'",
          },
        },
        required: ["location"],
      },
    },
  },
];

// Tool registry: maps tool name to implementation
const TOOL_REGISTRY: Record<string, (...args: any[]) => string> = {
  say_hello: sayHello,
  get_weather: getWeather,
};

async function chat() {
  const client = new OpenAI({
    apiKey: process.env.OPENROUTER_API_KEY!,
    baseURL: "https://openrouter.ai/api/v1",
  });

  const MODEL = "deepseek/deepseek-v4-flash";

  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a helpful assistant." },
  ];

  while (true) {
    const userInput = prompt("You>> ");
    if (!userInput || ["exit", "quit"].includes(userInput.toLowerCase())) {
      break;
    }

    messages.push({ role: "user", content: userInput });

    while (true) {
      const response = await client.chat.completions.create({
        model: MODEL,
        messages,
        tools: TOOL_DEFINITIONS,
      });

      const msg = response.choices[0].message;
      messages.push(msg);

      if (msg.content) {
        console.log(`Assistant>> ${msg.content}`);
      }

      if (!msg.tool_calls) {
        break;
      }

      for (const toolCall of msg.tool_calls as ToolCall[]) {
        const toolName = toolCall.function.name;
        const toolArgs = JSON.parse(toolCall.function.arguments);

        if (toolName in TOOL_REGISTRY) {
          const toolFunc = TOOL_REGISTRY[toolName];
          const result = toolFunc(...Object.values(toolArgs));
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: result,
          });
        } else {
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: `Tool ${toolName} not found.`,
          });
        }
      }
    }
  }
}

chat();
