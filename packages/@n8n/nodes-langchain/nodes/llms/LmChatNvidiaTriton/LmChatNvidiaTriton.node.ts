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

// LangChain Core Imports
import {
	BaseChatModel,
	type BaseChatModelParams,
} from '@langchain/core/language_models/chat_models';
import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { type ChatResult } from '@langchain/core/outputs';
import { type CallbackManagerForLLMRun, type Callbacks } from '@langchain/core/callbacks/manager';
import { v4 as uuidv4 } from 'uuid'; // For generating unique IDs

// --- Custom ChatNvidiaTriton LangChain Model (for /infer endpoint) ---
interface ChatNvidiaTritonInferParams extends BaseChatModelParams {
	tritonBaseUrl: string;
	modelName: string;
	inputFieldName: string;
	outputFieldName: string;
	temperature?: number; // Will be passed in 'parameters' if Triton model supports it via /infer
	// Note: The /infer endpoint is more generic. Passing arbitrary parameters like temperature
	// depends on the specific model's configuration in Triton and if it accepts them via
	// the "parameters" field in the request payload. The provided cURL doesn't show it.
	n8nHttpRequest: IExecuteFunctions['helpers']['httpRequest'];
	onFailedAttempt?: (error: Error) => void;
}

class ChatNvidiaTritonInfer extends BaseChatModel {
	private tritonBaseUrl: string;
	private modelName: string;
	private inputFieldName: string;
	private outputFieldName: string;
	private temperature?: number;
	private n8nHttpRequest: IExecuteFunctions['helpers']['httpRequest'];
	private onFailedAttemptHandler?: (error: Error) => void;

	constructor(fields: ChatNvidiaTritonInferParams) {
		super(fields);
		this.tritonBaseUrl = fields.tritonBaseUrl.replace(/\/+$/, '');
		this.modelName = fields.modelName;
		this.inputFieldName = fields.inputFieldName;
		this.outputFieldName = fields.outputFieldName;
		this.temperature = fields.temperature; // Store if provided
		this.n8nHttpRequest = fields.n8nHttpRequest;
		this.onFailedAttemptHandler = fields.onFailedAttempt;
		this.callbacks = fields.callbacks;
	}

	_llmType(): string {
		return 'chat_nvidia_triton_infer';
	}

	async _generate(
		messages: BaseMessage[],
		options: this['ParsedCallOptions'],
		runManager?: CallbackManagerForLLMRun,
	): Promise<ChatResult> {
		let promptText = ' ';
		if (messages.length > 0) {
			const lastMessage = messages[messages.length - 1];
			if (typeof lastMessage.content === 'string') {
				promptText = lastMessage.content;
			} else {
				promptText = JSON.stringify(lastMessage.content);
				console.warn(
					`[ChatNvidiaTritonInfer] Last message content was not a string, used JSON.stringify. Input: ${promptText}`,
				);
			}
		}

		const requestId = `n8n_${uuidv4()}`; // Generate a unique ID for the request

		const payload: JsonObject = {
			id: requestId,
			inputs: [
				{
					name: this.inputFieldName,
					shape: [1], // Assuming single string input
					datatype: 'BYTES', // Standard for text input with vLLM backend
					data: [promptText],
				},
			],
			outputs: [
				{
					name: this.outputFieldName,
					// parameters: { "binary_data_output": false } // Some backends might need this for string output
				},
			],
		};

		// Add optional parameters if the model supports them (e.g., temperature)
		// This is model-specific for the /infer endpoint.
		// The cURL example did not include this, but it's a common LLM parameter.
		// Check your Triton model's documentation for supported parameters.
		if (this.temperature !== undefined) {
			payload.parameters = {
				// This 'parameters' object is at the root of the payload
				temperature: this.temperature,
				// "stream": false, // if your model supports a stream parameter here
			};
		}

		const endpoint = `${this.tritonBaseUrl}/v2/models/${this.modelName}/infer`;
		let responseData: JsonObject;

		try {
			responseData = (await this.n8nHttpRequest({
				method: 'POST',
				body: payload,
				headers: { 'Content-Type': 'application/json' },
				url: endpoint,
				json: true,
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
			(errorToThrow as any).status = status;
			throw errorToThrow;
		}

		if (!responseData.outputs || !Array.isArray(responseData.outputs)) {
			throw new Error(
				`Invalid response structure from Triton: 'outputs' array not found. Response: ${JSON.stringify(
					responseData,
				)}`,
			);
		}

		const outputObject = responseData.outputs.find((o: any) => o.name === this.outputFieldName);

		if (!outputObject) {
			throw new Error(
				`Output field '${this.outputFieldName}' not found in Triton response. Available outputs: ${responseData.outputs
					.map((o: any) => o.name)
					.join(', ')}. Response: ${JSON.stringify(responseData)}`,
			);
		}

		// Assuming the data is in outputObject.data[0] and is a string
		// For BYTES datatype, Triton often returns an array of strings.
		if (!outputObject.data || !Array.isArray(outputObject.data) || outputObject.data.length === 0) {
			throw new Error(
				`Data for output field '${this.outputFieldName}' is missing or not an array in Triton response. Output object: ${JSON.stringify(outputObject)}`,
			);
		}

		const textOutput = outputObject.data[0] as string;

		if (typeof textOutput !== 'string') {
			throw new Error(
				`Expected string data for output field '${
					this.outputFieldName
				}', but received type ${typeof textOutput}. Output data: ${JSON.stringify(outputObject.data)}`,
			);
		}

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
				id: responseData.id,
				raw_response: responseData,
			},
		};
	}
}
// --- End of Custom ChatNvidiaTriton LangChain Model ---

export class LmChatNvidiaTriton implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Nvidia Triton Model (Infer)',
		name: 'lmChatNvidiaTritonInfer', // Updated name
		icon: 'file:nvidiaTriton.svg',
		group: ['transform'],
		version: 1, // Consider incrementing if this is a major change from a previous version
		description: 'Uses Nvidia Triton Inference Server /infer endpoint',
		defaults: {
			name: 'Nvidia Triton (Infer)',
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
				name: 'nvidiaTritonApi',
				required: true,
			},
		],
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiChain, NodeConnectionTypes.AiAgent]),
			{
				displayName: 'Model Name',
				name: 'modelName',
				type: 'string',
				required: true,
				default: 'vllm_qwen_qwq', // From your new cURL
				description:
					'The name of the model deployed on Triton (e.g., "vllm_qwen_qwq" for /v2/models/vllm_qwen_qwq/infer)',
			},
			{
				displayName: 'Input Field Name',
				name: 'inputFieldName',
				type: 'string',
				required: true,
				default: 'INPUT', // From your new cURL
				description: 'The "name" of the input tensor in the Triton request payload.',
			},
			{
				displayName: 'Output Field Name',
				name: 'outputFieldName',
				type: 'string',
				required: true,
				default: 'OUTPUT', // From your new cURL
				description: 'The "name" of the output tensor to retrieve from the Triton response.',
			},
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				description:
					'Optional parameters for the /infer endpoint. Support depends on the Triton model.',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Temperature',
						name: 'temperature',
						default: 0.7, // A common default, but your cURL didn't specify it for /infer
						typeOptions: { maxValue: 2.0, minValue: 0.0, numberPrecision: 2 },
						description:
							'Controls randomness. Note: Support for this parameter via /infer depends on the specific Triton model configuration.',
						type: 'number',
					},
					// Add other parameters if your model supports them via the root "parameters" object
					// in the /infer request.
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials('nvidiaTritonApi');
		const tritonBaseUrl = (credentials.baseUrl as string) || 'http://localhost:8000';

		const modelName = this.getNodeParameter('modelName', itemIndex) as string;
		const inputFieldName = this.getNodeParameter('inputFieldName', itemIndex) as string;
		const outputFieldName = this.getNodeParameter('outputFieldName', itemIndex) as string;

		const options = this.getNodeParameter('options', itemIndex, {}) as {
			temperature?: number;
		};

		const executeFunctionsThis = this as unknown as IExecuteFunctions;

		const model = new ChatNvidiaTritonInfer({
			tritonBaseUrl,
			modelName,
			inputFieldName,
			outputFieldName,
			temperature: options.temperature,
			callbacks: [new N8nLlmTracing(executeFunctionsThis)],
			onFailedAttempt: makeN8nLlmFailedAttemptHandler(executeFunctionsThis),
			n8nHttpRequest: executeFunctionsThis.helpers.httpRequest,
		});

		return {
			response: model,
		};
	}
}
