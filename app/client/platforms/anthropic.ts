import { Anthropic, ApiPath } from "@/app/constant";
import { ChatOptions, getHeaders, LLMApi, SpeechOptions } from "../api";
import {
  useAccessStore,
  useAppConfig,
  useChatStore,
  usePluginStore,
  ChatMessageTool,
} from "@/app/store";
import { getClientConfig } from "@/app/config/client";
import { ANTHROPIC_BASE_URL } from "@/app/constant";
import { getMessageTextContent, isVisionModel } from "@/app/utils";
import { preProcessImageContent, streamWithThink } from "@/app/utils/chat";
import { cloudflareAIGatewayUrl } from "@/app/utils/cloudflare";
import { RequestPayload } from "./openai";
import { fetch } from "@/app/utils/stream";

export type MultiBlockContent = {
  type: "image" | "text";
  source?: {
    type: string;
    media_type: string;
    data: string;
  };
  text?: string;
};

export type AnthropicMessage = {
  role: (typeof ClaudeMapper)[keyof typeof ClaudeMapper];
  content: string | MultiBlockContent[];
};

export interface AnthropicChatRequest {
  model: string; // The model that will complete your prompt.
  messages: AnthropicMessage[]; // The prompt that you want Claude to complete.
  max_tokens: number; // The maximum number of tokens to generate before stopping.
  stop_sequences?: string[]; // Sequences that will cause the model to stop generating completion text.
  temperature?: number; // Amount of randomness injected into the response.
  top_p?: number; // Use nucleus sampling.
  top_k?: number; // Only sample from the top K options for each subsequent token.
  metadata?: object; // An object describing metadata about the request.
  stream?: boolean; // Whether to incrementally stream the response using server-sent events.
  system?: string; // System prompt (must be top-level, not a "user" message).
  thinking?:
    | { type: "adaptive"; display?: "summarized" | "omitted" }
    | { type: "enabled"; budget_tokens: number; display?: "summarized" };
  output_config?: { effort: string };
}

export interface ChatRequest {
  model: string; // The model that will complete your prompt.
  prompt: string; // The prompt that you want Claude to complete.
  max_tokens_to_sample: number; // The maximum number of tokens to generate before stopping.
  stop_sequences?: string[]; // Sequences that will cause the model to stop generating completion text.
  temperature?: number; // Amount of randomness injected into the response.
  top_p?: number; // Use nucleus sampling.
  top_k?: number; // Only sample from the top K options for each subsequent token.
  metadata?: object; // An object describing metadata about the request.
  stream?: boolean; // Whether to incrementally stream the response using server-sent events.
}

export interface ChatResponse {
  completion: string;
  stop_reason: "stop_sequence" | "max_tokens";
  model: string;
}

export type ChatStreamResponse = ChatResponse & {
  stop?: string;
  log_id: string;
};

const ClaudeMapper = {
  assistant: "assistant",
  user: "user",
  system: "user",
} as const;


/**
 * Claude 4.7+ / 5.x models (Sonnet 5.x, Opus 5.x, Fable, Mythos ...):
 *  - only support adaptive thinking (`type: "enabled"` + budget_tokens => 400)
 *  - reject non-default temperature / top_p / top_k (=> 400)
 */
export function isAdaptiveOnlyClaude(model: string) {
  return /claude-(?:opus|sonnet|haiku|fable|mythos)-(?:4-(?:[7-9]|1\d)(?!\d)|(?:[5-9]|\d{2,})(?!\d{4}))/i.test(
    model,
  );
}

/** Legacy models that can think with a manual token budget. */
export function isLegacyThinkingClaude(model: string) {
  return /claude-(?:3-7-sonnet|(?:sonnet|opus)-4(?:-[0-6])?(?:-\d{8})?$|(?:sonnet|opus|haiku)-4-[56])/i.test(
    model,
  );
}

const LEGACY_THINKING_BUDGET: Record<string, number> = {
  low: 2048,
  medium: 4096,
  high: 10000,
  xhigh: 16000,
  max: 24000,
};

const keys = ["claude-2, claude-instant-1"];

export class ClaudeApi implements LLMApi {
  speech(options: SpeechOptions): Promise<ArrayBuffer> {
    throw new Error("Method not implemented.");
  }

  extractMessage(res: any) {
    console.log("[Response] claude response: ", res);

    if (res?.error) return `[Anthropic API error: ${res.error.message}]`;
    return (res?.content ?? [])
      .filter((b: any) => b?.type === "text")
      .map((b: any) => b.text)
      .join("");
  }
  async chat(options: ChatOptions): Promise<void> {
    const visionModel = isVisionModel(options.config.model);

    const accessStore = useAccessStore.getState();

    const shouldStream = !!options.config.stream;

    const modelConfig = {
      ...useAppConfig.getState().modelConfig,
      ...useChatStore.getState().currentSession().mask.modelConfig,
      ...{
        model: options.config.model,
      },
    };

    // try get base64image from local cache image_url
    // System prompts (mask context, memory summary, MCP prompt ...) are sent in
    // Anthropic's top-level `system` field instead of being faked as user turns.
    const systemTexts: string[] = [];
    const messages: ChatOptions["messages"] = [];
    for (const v of options.messages) {
      if (v.role === "system") {
        const t = getMessageTextContent(v).trim();
        if (t) systemTexts.push(t);
        continue;
      }
      const content = await preProcessImageContent(v.content);
      messages.push({ role: v.role, content });
    }
    const systemPrompt = systemTexts.join("\n\n");

    // thinking is shown as a "> quoted" block at the top of assistant messages,
    // never send it back to the model as if it were part of the answer
    const stripThinking = (text: string) => {
      const lines = text.split("\n");
      let i = 0;
      while (i < lines.length && (lines[i].startsWith(">") || !lines[i].trim()))
        i++;
      let end = lines.length;
      while (end > i && (lines[end - 1].startsWith(">") || !lines[end - 1].trim()))
        end--;
      return (i > 0 || end < lines.length) && i < end
        ? lines.slice(i, end).join("\n")
        : text;
    };
    const textOf = (v: (typeof messages)[number]) => {
      const text = getMessageTextContent(v);
      return v.role === "assistant" ? stripThinking(text) : text;
    };

    const prompt = messages
      .filter((v) => {
        if (!v.content) return false;
        if (typeof v.content === "string" && !v.content.trim()) return false;
        return true;
      })
      .map((v) => {
        const { role, content } = v;
        const insideRole = ClaudeMapper[role] ?? "user";

        if (!visionModel || typeof content === "string") {
          return {
            role: insideRole,
            content: textOf(v),
          };
        }
        return {
          role: insideRole,
          content: content
            .filter((v) => v.image_url || v.text)
            .map(({ type, text, image_url }) => {
              if (type === "text") {
                return {
                  type,
                  text: text!,
                };
              }
              const { url = "" } = image_url || {};
              const colonIndex = url.indexOf(":");
              const semicolonIndex = url.indexOf(";");
              const comma = url.indexOf(",");

              const mimeType = url.slice(colonIndex + 1, semicolonIndex);
              const encodeType = url.slice(semicolonIndex + 1, comma);
              const data = url.slice(comma + 1);

              return {
                type: "image" as const,
                source: {
                  type: encodeType,
                  media_type: mimeType,
                  data,
                },
              };
            }),
        };
      });

    // roles must alternate between "user" and "assistant" in claude
    const merged: typeof prompt = [];
    for (const m of prompt) {
      const last = merged[merged.length - 1];
      if (last && last.role === m.role) {
        merged.push({
          role: m.role === "user" ? "assistant" : "user",
          content: ";",
        });
      }
      merged.push(m);
    }
    prompt.splice(0, prompt.length, ...merged);

    if (prompt[0]?.role === "assistant") {
      prompt.unshift({
        role: "user",
        content: ";",
      });
    }

    // ---- sampling + thinking -------------------------------------------
    const model = modelConfig.model;
    const adaptiveOnly = isAdaptiveOnlyClaude(model);
    const legacyThinking = !adaptiveOnly && isLegacyThinkingClaude(model);
    const wantThinking = modelConfig.anthropicThinking ?? true;
    const effort = modelConfig.anthropicEffort ?? "high";
    const maxTokens = modelConfig.max_tokens;

    const requestBody: AnthropicChatRequest = {
      messages: prompt,
      stream: shouldStream,
      model,
      max_tokens: maxTokens,
    };
    if (systemPrompt) requestBody.system = systemPrompt;

    if (adaptiveOnly) {
      // Sonnet 5.x etc: temperature/top_p/top_k must NOT be sent at all.
      if (wantThinking) {
        requestBody.thinking = { type: "adaptive", display: "summarized" };
        requestBody.output_config = { effort };
      }
    } else if (legacyThinking && wantThinking && maxTokens > 2048) {
      // thinking requires temperature 1 and budget_tokens < max_tokens
      requestBody.thinking = {
        type: "enabled",
        budget_tokens: Math.min(
          LEGACY_THINKING_BUDGET[effort] ?? 10000,
          maxTokens - 1024,
        ),
      };
    } else {
      // newer legacy models reject temperature + top_p together, so only temperature
      requestBody.temperature = modelConfig.temperature;
    }

    const path = this.path(Anthropic.ChatPath);

    const controller = new AbortController();
    options.onController?.(controller);

    if (shouldStream) {
      let index = -1;
      // thinking blocks (with signatures) of the current turn. They must be sent
      // back to the API in front of tool_use blocks when a tool is called.
      let thinkingBlocks: any[] = [];
      let curBlock: any = null;
      let thinkingSeen = 0; // thinking blocks received in the current response
      const [tools, funcs] = usePluginStore
        .getState()
        .getAsTools(
          useChatStore.getState().currentSession().mask?.plugin || [],
        );
      return streamWithThink(
        path,
        requestBody,
        {
          ...getHeaders(),
          "anthropic-version": accessStore.anthropicApiVersion,
        },
        // @ts-ignore
        tools.map((tool) => ({
          name: tool?.function?.name,
          description: tool?.function?.description,
          input_schema: tool?.function?.parameters,
        })),
        funcs,
        controller,
        // parseSSE
        (text: string, runTools: ChatMessageTool[]) => {
          const chunkJson: any = JSON.parse(text);
          const none = { isThinking: false, content: "" };

          if (chunkJson?.type === "error") {
            const msg = chunkJson?.error?.message ?? text;
            return {
              isThinking: false,
              content: `\n\n[Anthropic API error: ${msg}]`,
            };
          }

          if (chunkJson?.type === "message_start") {
            thinkingSeen = 0;
            return none;
          }

          // End of one model response: add a small status line so it is
          // obvious whether thinking was requested and whether it happened.
          if (chunkJson?.type === "message_delta") {
            const reason = chunkJson?.delta?.stop_reason;
            const u = chunkJson?.usage ?? {};
            let status = "";
            if (reason !== "tool_use") {
              if (requestBody.thinking) {
                const eff = requestBody.output_config?.effort;
                status =
                  `\n\n> ⚙ thinking requested (${requestBody.thinking.type}` +
                  `${eff ? ", effort " + eff : ""}) · thinking blocks received: ${thinkingSeen}` +
                  ` · thinking tokens: ${
                    u.output_tokens_details?.thinking_tokens ?? "n/a"
                  } · output tokens: ${u.output_tokens ?? "n/a"}`;
              } else if (wantThinking) {
                status = `\n\n> ⚙ thinking was NOT sent: model name "${model}" is not recognised as a thinking model`;
              }
            }
            let notice = "";
            if (reason === "refusal") {
              notice =
                "\n\n[Assistant refused to respond. Please modify your request and try again.]";
              options.onError?.(
                new Error("Content policy violation: " + notice),
              );
            } else if (reason === "max_tokens") {
              notice = `\n\n[Cut off: hit the max output tokens limit (${maxTokens}, thinking included). Raise "Max Tokens" in THIS chat's settings.]`;
            }
            return { isThinking: false, content: notice + status };
          }

          const block = chunkJson?.content_block;
          if (chunkJson?.type === "content_block_start") {
            if (block?.type === "thinking") {
              thinkingSeen += 1;
              curBlock = { type: "thinking", thinking: "", signature: "" };
            } else if (block?.type === "redacted_thinking") {
              thinkingSeen += 1;
              thinkingBlocks.push(block);
            } else if (block?.type === "tool_use") {
              index += 1;
              runTools.push({
                id: block.id,
                type: "function",
                function: {
                  name: block.name,
                  arguments: "",
                },
              });
            }
            return none;
          }
          if (chunkJson?.type === "content_block_stop") {
            if (curBlock) {
              thinkingBlocks.push(curBlock);
              curBlock = null;
            }
            return none;
          }

          const delta = chunkJson?.delta;
          if (delta?.type === "thinking_delta") {
            if (curBlock) curBlock.thinking += delta.thinking ?? "";
            return { isThinking: true, content: delta.thinking ?? "" };
          }
          if (delta?.type === "signature_delta") {
            if (curBlock) curBlock.signature += delta.signature ?? "";
            return none;
          }
          if (delta?.type == "input_json_delta" && delta?.partial_json) {
            // @ts-ignore
            runTools[index]["function"]["arguments"] += delta.partial_json;
            return none;
          }
          if (delta?.type === "text_delta") {
            return { isThinking: false, content: delta.text ?? "" };
          }
          return none;
        },
        // processToolMessage, include tool_calls message and tool call results
        (
          requestPayload: RequestPayload,
          toolCallMessage: any,
          toolCallResult: any[],
        ) => {
          // reset index value
          index = -1;
          const priorThinking = thinkingBlocks;
          thinkingBlocks = [];
          // @ts-ignore
          requestPayload?.messages?.splice(
            // @ts-ignore
            requestPayload?.messages?.length,
            0,
            {
              role: "assistant",
              content: [
                ...priorThinking,
                ...toolCallMessage.tool_calls.map((tool: ChatMessageTool) => ({
                  type: "tool_use",
                  id: tool.id,
                  name: tool?.function?.name,
                  input: tool?.function?.arguments
                    ? JSON.parse(tool?.function?.arguments)
                    : {},
                })),
              ],
            },
            // @ts-ignore
            ...toolCallResult.map((result) => ({
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: result.tool_call_id,
                  content: result.content,
                },
              ],
            })),
          );
        },
        options,
      );
    } else {
      const payload = {
        method: "POST",
        body: JSON.stringify(requestBody),
        signal: controller.signal,
        headers: {
          ...getHeaders(), // get common headers
          "anthropic-version": accessStore.anthropicApiVersion,
          // do not send `anthropicApiKey` in browser!!!
          // Authorization: getAuthKey(accessStore.anthropicApiKey),
        },
      };

      try {
        controller.signal.onabort = () =>
          options.onFinish("", new Response(null, { status: 400 }));

        const res = await fetch(path, payload);
        const resJson = await res.json();

        const message = this.extractMessage(resJson);
        options.onFinish(message, res);
      } catch (e) {
        console.error("failed to chat", e);
        options.onError?.(e as Error);
      }
    }
  }
  async usage() {
    return {
      used: 0,
      total: 0,
    };
  }
  async models() {
    // const provider = {
    //   id: "anthropic",
    //   providerName: "Anthropic",
    //   providerType: "anthropic",
    // };

    return [
      // {
      //   name: "claude-instant-1.2",
      //   available: true,
      //   provider,
      // },
      // {
      //   name: "claude-2.0",
      //   available: true,
      //   provider,
      // },
      // {
      //   name: "claude-2.1",
      //   available: true,
      //   provider,
      // },
      // {
      //   name: "claude-3-opus-20240229",
      //   available: true,
      //   provider,
      // },
      // {
      //   name: "claude-3-sonnet-20240229",
      //   available: true,
      //   provider,
      // },
      // {
      //   name: "claude-3-haiku-20240307",
      //   available: true,
      //   provider,
      // },
    ];
  }
  path(path: string): string {
    const accessStore = useAccessStore.getState();

    let baseUrl: string = "";

    if (accessStore.useCustomConfig) {
      baseUrl = accessStore.anthropicUrl;
    }

    // if endpoint is empty, use default endpoint
    if (baseUrl.trim().length === 0) {
      const isApp = !!getClientConfig()?.isApp;

      baseUrl = isApp ? ANTHROPIC_BASE_URL : ApiPath.Anthropic;
    }

    if (!baseUrl.startsWith("http") && !baseUrl.startsWith("/api")) {
      baseUrl = "https://" + baseUrl;
    }

    baseUrl = trimEnd(baseUrl, "/");

    // try rebuild url, when using cloudflare ai gateway in client
    return cloudflareAIGatewayUrl(`${baseUrl}/${path}`);
  }
}

function trimEnd(s: string, end = " ") {
  if (end.length === 0) return s;

  while (s.endsWith(end)) {
    s = s.slice(0, -end.length);
  }

  return s;
}
