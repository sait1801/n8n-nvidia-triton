/* eslint-disable n8n-nodes-base/node-dirname-against-convention */
import {
	NodeConnectionTypes,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
	type IExecuteFunctions,
	type JsonObject,
	type NodeApiError,
} from 'n8n-workflow';

// Assuming these utility paths are correct in your n8n development environment
import { getHttpProxyAgent } from '@utils/httpProxyAgent';
import { getConnectionHintNoticeField } from '@utils/sharedFields';
import { makeN8nLlmFailedAttemptHandler } from '../n8nLlmFailedAttemptHandler'; // Adjust path as needed
import { N8nLlmTracing } from '../N8nLlmTracing'; // Adjust path as needed

// LangChain Core Imports for the custom ChatNvidiaTriton class
import {
	BaseChatModel,
	type BaseChatModelParams,
} from '@langchain/core/language_models/chat_models';
import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { type ChatResult } from '@langchain/core/outputs';
import { type CallbackManagerForLLMRun, type Callbacks } from '@langchain/core/callbacks/manager';
import { type Agent } from 'undici'; // For httpAgent type (though n8n's httpRequest handles agent internally)

// --- Custom ChatNvidiaTriton LangChain Model ---
interface ChatNvidiaTritonParams extends BaseChatModelParams {
	tritonBaseUrl: string;
	modelName: string;
	temperature?: number;
	stream?: boolean;
	// This helper is crucial for making HTTP requests via n8n's infrastructure
	n8nHttpRequest: IExecuteFunctions['helpers']['httpRequest'];
	onFailedAttempt?: (error: Error) => void;
}

class ChatNvidiaTriton extends BaseChatModel {
	private tritonBaseUrl: string;
	private modelName: string;
	private temperature: number;
	private streamOutput: boolean;
	private n8nHttpRequest: IExecuteFunctions['helpers']['httpRequest'];
	private onFailedAttemptHandler?: (error: Error) => void;

	constructor(fields: ChatNvidiaTritonParams) {
		super(fields);
		this.tritonBaseUrl = fields.tritonBaseUrl.replace(/\/+$/, ''); // Ensure no trailing slash
		this.modelName = fields.modelName;
		this.temperature = fields.temperature ?? 0.0;
		this.streamOutput = fields.stream ?? false;
		this.n8nHttpRequest = fields.n8nHttpRequest;
		this.onFailedAttemptHandler = fields.onFailedAttempt;
		this.callbacks = fields.callbacks; // Pass callbacks to BaseChatModel
	}

	_llmType(): string {
		return 'chat_nvidia_triton_generate_simplified';
	}

	async _generate(
		messages: BaseMessage[],
		options: this['ParsedCallOptions'], // Contains stop sequences, etc.
		runManager?: CallbackManagerForLLMRun,
	): Promise<ChatResult> {
		// For Triton's /generate, we typically need a single text input.
		// We'll use the content of the last message.
		let promptText = ' '; // Default to a space if no messages or content
		if (messages.length > 0) {
			const lastMessage = messages[messages.length - 1];
			if (typeof lastMessage.content === 'string') {
				promptText = lastMessage.content;
			} else {
				// If content is not a simple string (e.g., list of content blocks),
				// try to serialize it or extract relevant text.
				// For simplicity here, we'll just stringify.
				promptText = JSON.stringify(lastMessage.content);
				console.warn(
					`[ChatNvidiaTriton] Last message content was not a string, used JSON.stringify. Input: ${promptText}`,
				);
			}
		}

		const payload: JsonObject = {
			text_input: promptText,
			parameters: {
				stream: this.streamOutput,
				temperature: this.temperature,
				// You could add other parameters here if your Triton model supports them
				// e.g., max_tokens, stop_sequences (options.stop might be relevant)
			},
		};

		const endpoint = `${this.tritonBaseUrl}/v2/models/${this.modelName}/generate`;
		let responseData: JsonObject;

		try {
			responseData = (await this.n8nHttpRequest({
				method: 'POST',
				body: payload,
				headers: { 'Content-Type': 'application/json' },
				url: endpoint,
				json: true, // Expect JSON response
			})) as JsonObject;
		} catch (error) {
			const typedError = error as NodeApiError | Error;
			if (this.onFailedAttemptHandler) {
				this.onFailedAttemptHandler(typedError);
			}
			const errorMessage =
				(typedError as NodeApiError).cause?.message ||
				typedError.message ||
				'Unknown error during Triton API call';
			const status =
				(typedError as NodeApiError).cause?.response?.status ||
				(typedError as NodeApiError).httpCode ||
				500;
			const errorToThrow = new Error(`Nvidia Triton API Error (Status ${status}): ${errorMessage}`);
			(errorToThrow as any).status = status; // Langchain might check for status
			throw errorToThrow;
		}

		const textOutput = responseData.text_output as string | undefined;

		if (typeof textOutput !== 'string') {
			throw new Error(
				`Invalid response structure from Triton: 'text_output' not found or not a string. Response: ${JSON.stringify(
					responseData,
				)}`,
			);
		}

		// If streaming were fully handled here (it's not for /generate in this simple model),
		// runManager.handleLLMNewToken would be called multiple times.
		// For a non-streaming /generate endpoint, we call it once with the full output.
		await runManager?.handleLLMNewToken(textOutput);

		return {
			generations: [
				{
					text: textOutput,
					message: new AIMessage(textOutput),
				},
			],
			llmOutput: {
				model_name: responseData.model_name || this.modelName,
				model_version: responseData.model_version,
				raw_response: responseData,
			},
		};
	}
}
// --- End of Custom ChatNvidiaTriton LangChain Model ---

export class LmChatNvidiaTriton implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Nvidia Triton Model (Simple)',
		name: 'lmChatNvidiaTriton', // Changed name slightly to avoid conflict if you have the other one
		icon: 'file:nvidiaTriton.svg', // You'll need to create/add this SVG icon
		group: ['transform'],
		version: 1,
		description: 'Uses Nvidia Triton Inference Server /generate endpoint',
		defaults: {
			name: 'Nvidia Triton',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Language Models', 'Root Nodes'],
				'Language Models': ['Chat Models (Recommended)'],
			},
			resources: {
				// primaryDocumentation: [ { url: 'YOUR_TRITON_DOCS_LINK_HERE' } ],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiLanguageModel],
		outputNames: ['Model'],
		credentials: [
			{
				name: 'nvidiaTritonApi', // This credential type needs to be defined in n8n
				required: true,
				// It should provide a 'baseUrl' field, e.g., "http://localhost:8000"
			},
		],
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiChain, NodeConnectionTypes.AiAgent]),
			{
				displayName: 'Model Name',
				name: 'modelName',
				type: 'string',
				required: true,
				default: 'llama2vllm', // From your cURL example
				description:
					'The name of the model deployed on Triton (e.g., "llama2vllm" for /v2/models/llama2vllm/generate)',
			},
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				description: 'Parameters for the /generate endpoint',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Temperature',
						name: 'temperature',
						default: 0.0, // From your cURL
						typeOptions: { maxValue: 2.0, minValue: 0.0, numberPrecision: 2 },
						description: 'Controls randomness. 0.0 is deterministic.',
						type: 'number',
					},
					{
						displayName: 'Stream (Parameter)',
						name: 'stream',
						default: false, // From your cURL
						description:
							'Sets the "stream" parameter for Triton. Note: This node provides the full response, not chunked streaming output to n8n.',
						type: 'boolean',
					},
					// Add other parameters like 'max_tokens' if your Triton model's
					// /generate endpoint supports them within its "parameters" object.
					// {
					// displayName: 'Max Tokens',
					// name: 'max_tokens', // Ensure this matches Triton's expected parameter name
					// type: 'number',
					// default: 1024,
					// description: 'Maximum number of tokens to generate.',
					// },
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials('nvidiaTritonApi');
		// Ensure baseUrl is provided by the credential, with a fallback (though credential should enforce it)
		const tritonBaseUrl = (credentials.baseUrl as string) || 'http://localhost:8000';

		const modelName = this.getNodeParameter('modelName', itemIndex) as string;
		const options = this.getNodeParameter('options', itemIndex, {}) as {
			temperature?: number;
			stream?: boolean;
			// max_tokens?: number; // if you add it to properties
		};

		// Cast 'this' to IExecuteFunctions where n8n utilities expect it.
		// This is a common pattern in n8n node development when ISupplyDataFunctions
		// needs to use helpers typically available on IExecuteFunctions.
		const executeFunctionsThis = this as unknown as IExecuteFunctions;

		const model = new ChatNvidiaTriton({
			tritonBaseUrl,
			modelName,
			temperature: options.temperature,
			stream: options.stream,
			// max_tokens: options.max_tokens, // if you add it
			callbacks: [new N8nLlmTracing(executeFunctionsThis)],
			// getHttpProxyAgent is used by n8n's internal httpRequest, so we don't pass an agent directly to ChatNvidiaTriton
			// The proxy configuration is handled globally for n8n's httpRequest.
			// httpAgent: getHttpProxyAgent(), // Not directly used by the simplified ChatNvidiaTriton
			onFailedAttempt: makeN8nLlmFailedAttemptHandler(executeFunctionsThis),
			n8nHttpRequest: executeFunctionsThis.helpers.httpRequest, // Pass the helper directly
		});

		return {
			response: model,
		};
	}
}
