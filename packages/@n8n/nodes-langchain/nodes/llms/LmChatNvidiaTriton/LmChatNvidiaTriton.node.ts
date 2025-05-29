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

import { getConnectionHintNoticeField } from '@utils/sharedFields';
import { makeN8nLlmFailedAttemptHandler } from '../n8nLlmFailedAttemptHandler'; // Adjust path as needed
import { N8nLlmTracing } from '../N8nLlmTracing'; // Adjust path as needed

// LangChain Core Imports
import {
	BaseChatModel,
	type BaseChatModelParams,
} from '@langchain/core/language_models/chat_models';
import { AIMessage, BaseMessage, type LanguageModelInput } from '@langchain/core/messages';
import { type ChatResult } from '@langchain/core/outputs';
import { type CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import { v4 as uuidv4 } from 'uuid';
import type { Runnable } from '@langchain/core/runnables';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { ZodObject } from 'zod';

// --- Custom ChatNvidiaTriton LangChain Model (for /infer endpoint) ---
interface ChatNvidiaTritonParams extends BaseChatModelParams {
	tritonBaseUrl: string;
	modelName: string;
	inputFieldName: string;
	outputFieldName: string;
	temperature?: number;
	n8nHttpRequest: IExecuteFunctions['helpers']['httpRequest'];
	onFailedAttempt?: (error: Error) => void;
}

class ChatNvidiaTriton extends BaseChatModel {
	private tritonBaseUrl: string;
	private modelName: string;
	private inputFieldName: string;
	private outputFieldName: string;
	private temperature?: number;
	private n8nHttpRequest: IExecuteFunctions['helpers']['httpRequest'];
	private onFailedAttemptHandler?: (error: Error) => void;

	public _isChatModel = true; // Explicitly set

	constructor(fields: ChatNvidiaTritonParams) {
		super(fields);
		this.tritonBaseUrl = fields.tritonBaseUrl.replace(/\/+$/, '');
		this.modelName = fields.modelName;
		this.inputFieldName = fields.inputFieldName;
		this.outputFieldName = fields.outputFieldName;
		this.temperature = fields.temperature;
		this.n8nHttpRequest = fields.n8nHttpRequest;
		this.onFailedAttemptHandler = fields.onFailedAttempt;
		console.log(
			'[ChatNvidiaTriton] Instance created. Has bindTools:',
			typeof this.bindTools === 'function',
		);
	}

	_llmType(): string {
		return 'chat_nvidia_triton_infer';
	}

	// DRASTICALLY SIMPLIFIED bindTools
	bindTools(
		tools: (StructuredToolInterface | Record<string, unknown>)[],
		kwargs?: Partial<this['ParsedCallOptions']>,
	): Runnable<LanguageModelInput, AIMessage> {
		console.log(
			'[ChatNvidiaTriton] DRASTICALLY SIMPLIFIED bindTools CALLED. Tools:',
			tools,
			'kwargs:',
			kwargs,
		);
		// This version bypasses super.bindTools() and returns the current model instance.
		// This might prevent errors if the agent repeatedly calls bindTools on the returned object,
		// expecting it to always be a model instance.
		// WARNING: This means LangChain's default tool processing and binding to options
		// (which populates options.tools for _generate) is SKIPPED.
		// For a model that ignores tools anyway, this might be acceptable to avoid crashes.

		// If you wanted to try and manually put tools into lc_kwargs (part of what super.bindTools does):
		// this.lc_kwargs = { ...this.lc_kwargs, ...kwargs, tools: tools.map(convertToChatTool) };
		// However, this is risky without understanding all of super.bindTools() internals.
		// For now, just return 'this'.
		console.log('[ChatNvidiaTriton] DRASTIC bindTools returning THIS model instance.');
		return this;
	}

	async _generate(
		messages: BaseMessage[],
		options: this['ParsedCallOptions'],
		runManager?: CallbackManagerForLLMRun,
	): Promise<ChatResult> {
		// console.log('[ChatNvidiaTriton] _generate called. Options:', options);
		// Since we bypassed super.bindTools(), options.tools might not be populated here
		// in the way LangChain agents usually expect.
		// console.log('[ChatNvidiaTriton] Tools available in options for _generate:', options.tools);

		let promptText = ' ';
		if (messages.length > 0) {
			const lastMessage = messages[messages.length - 1];
			if (typeof lastMessage.content === 'string') {
				promptText = lastMessage.content;
			} else {
				promptText = JSON.stringify(lastMessage.content);
				console.warn(
					`[ChatNvidiaTriton] Last message content was not a string, used JSON.stringify. Input: ${promptText}`,
				);
			}
		}

		const requestId = `n8n_${uuidv4()}`;
		const payload: JsonObject = {
			id: requestId,
			inputs: [
				{
					name: this.inputFieldName,
					shape: [1],
					datatype: 'BYTES',
					data: [promptText],
				},
			],
			outputs: [{ name: this.outputFieldName }],
		};
		if (this.temperature !== undefined) {
			payload.parameters = { temperature: this.temperature };
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
			if (this.onFailedAttemptHandler) this.onFailedAttemptHandler(typedError);
			const errorMessage =
				(typedError as NodeApiError).cause?.message || typedError.message || 'Unknown error';
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
				`Invalid response: 'outputs' array not found. Response: ${JSON.stringify(responseData)}`,
			);
		}
		const outputObject = responseData.outputs.find((o: any) => o.name === this.outputFieldName);
		if (!outputObject) {
			throw new Error(
				`Output field '${this.outputFieldName}' not found. Available: ${responseData.outputs.map((o: any) => o.name).join(', ')}`,
			);
		}
		if (!outputObject.data || !Array.isArray(outputObject.data) || outputObject.data.length === 0) {
			throw new Error(
				`Data for '${this.outputFieldName}' missing/not array. Output: ${JSON.stringify(outputObject)}`,
			);
		}
		const textOutput = outputObject.data[0] as string;
		if (typeof textOutput !== 'string') {
			throw new Error(
				`Expected string data for '${this.outputFieldName}', got ${typeof textOutput}`,
			);
		}

		await runManager?.handleLLMNewToken(textOutput);
		const aiMessage = new AIMessage({ content: textOutput });

		return {
			generations: [{ text: textOutput, message: aiMessage }],
			llmOutput: {
				model_name: responseData.model_name || this.modelName,
				model_version: responseData.model_version,
				id: responseData.id || requestId,
				raw_response: responseData,
			},
		};
	}

	withStructuredOutput<RunOutput extends Record<string, any> = Record<string, any>>(
		schema: ZodObject<any, any, any, RunOutput> | Record<string, any>,
		config?: { name?: string; method?: 'functionCalling' | 'jsonMode'; includeRaw?: boolean },
	): Runnable<LanguageModelInput, RunOutput> {
		// console.log('[ChatNvidiaTriton] withStructuredOutput called with schema:', schema, 'config:', config);
		if (config?.method === 'jsonMode') {
			console.warn('[ChatNvidiaTriton] True JSON mode for withStructuredOutput is not supported.');
		}
		// eslint-disable-next-line @typescript-eslint/no-this-alias
		const llm = this;
		const runnable = {
			async invoke(input: LanguageModelInput, options?: Partial<any>): Promise<RunOutput> {
				const result = await llm.invoke(input, options);
				if (typeof result.content === 'string') {
					try {
						return JSON.parse(result.content) as RunOutput;
					} catch (e) {
						return { output: result.content } as unknown as RunOutput;
					}
				}
				return result.content as unknown as RunOutput;
			},
		};
		return runnable as Runnable<LanguageModelInput, RunOutput>;
	}
}

export class LmChatNvidiaTriton implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Nvidia Triton Model (Infer)',
		name: 'lmChatNvidiaTriton', // Ensure this matches the name used in any previous fixes
		icon: 'file:nvidiaTriton.svg',
		group: ['transform'],
		version: 1.2, // Incrementing version due to significant workaround
		description: 'Uses Nvidia Triton Inference Server /infer endpoint (with AI Agent workarounds)',
		defaults: { name: 'Nvidia Triton (Infer)' },
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Language Models', 'Root Nodes'],
				'Language Models': ['Chat Models (Recommended)'],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiLanguageModel],
		outputNames: ['Model'],
		credentials: [{ name: 'nvidiaTritonApi', required: true }],
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiChain, NodeConnectionTypes.AiAgent]),
			{
				displayName: 'Model Name',
				name: 'modelName',
				type: 'string',
				required: true,
				default: 'vllm_qwen_qwq',
				description: 'The name of the model deployed on Triton (e.g., "vllm_qwen_qwq")',
			},
			{
				displayName: 'Input Field Name',
				name: 'inputFieldName',
				type: 'string',
				required: true,
				default: 'INPUT',
				description: 'The "name" of the input tensor in the Triton request payload.',
			},
			{
				displayName: 'Output Field Name',
				name: 'outputFieldName',
				type: 'string',
				required: true,
				default: 'OUTPUT',
				description: 'The "name" of the output tensor to retrieve from the Triton response.',
			},
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Temperature',
						name: 'temperature',
						default: 0.7,
						typeOptions: { maxValue: 2.0, minValue: 0.0, numberPrecision: 2 },
						type: 'number',
						description: 'Controls randomness. Support depends on the Triton model configuration.',
					},
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
		const options = this.getNodeParameter('options', itemIndex, {}) as { temperature?: number };
		const executeFunctionsThis = this as unknown as IExecuteFunctions;

		const modelInstance = new ChatNvidiaTriton({
			tritonBaseUrl,
			modelName,
			inputFieldName,
			outputFieldName,
			temperature: options.temperature,
			callbacks: [new N8nLlmTracing(executeFunctionsThis)],
			onFailedAttempt: makeN8nLlmFailedAttemptHandler(executeFunctionsThis),
			n8nHttpRequest: executeFunctionsThis.helpers.httpRequest,
		});

		console.log('[LmChatNvidiaTriton.supplyData] Model instance created:', modelInstance);
		console.log(
			'[LmChatNvidiaTriton.supplyData] Model instance has bindTools:',
			typeof modelInstance.bindTools === 'function',
		);

		return { response: modelInstance };
	}
}
